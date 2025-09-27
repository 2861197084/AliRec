"""基于共现的召回策略。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Iterator

import duckdb

from src.recall.base import Candidate, RecallStrategy
from src.recall.loader import RecallDataContext
from src.recall.utils import apply_time_decay


@dataclass
class CovisitationConfig:
    lookback_days: int = 7
    window_hours: int = 24
    max_neighbors: int = 100
    max_per_user: int = 150
    behavior_weights: dict[int, float] | None = None


class CovisitationStrategy(RecallStrategy):
    def __init__(
        self,
        context: RecallDataContext,
        cutoff: datetime,
        config: CovisitationConfig | None = None,
    ) -> None:
        super().__init__(name="covisitation")
        self.context = context
        self.cutoff = cutoff
        self.config = config or CovisitationConfig()
        self.behavior_weights = self.config.behavior_weights or {
            4: 1.0,
            3: 0.7,
            2: 0.5,
            1: 0.2,
        }

    def generate(self, user_ids: Iterable[int]) -> Iterator[Candidate]:
        user_id_list = list(user_ids)
        if not user_id_list:
            return iter([])

        lookback_start = self.cutoff - timedelta(days=self.config.lookback_days)
        con = self.context.connect()

        query = f"""
            WITH recent_behavior AS (
                SELECT user_id, item_id, behavior_type, time
                FROM read_parquet('{self.context.behavior_pattern}')
                WHERE time >= ? AND time < ?
                  AND user_id IN ({','.join('?' for _ in user_id_list)})
            ),
            labeled AS (
                SELECT
                    user_id,
                    item_id,
                    behavior_type,
                    time,
                    CASE behavior_type
                        {self._build_weight_case()}
                        ELSE 0.0
                    END AS base_weight
                FROM recent_behavior
            ),
            pairs AS (
                SELECT
                    a.user_id,
                    a.item_id AS anchor_item,
                    b.item_id AS neighbor_item,
                    GREATEST(a.time, b.time) AS recent_time,
                    ABS(EXTRACT(EPOCH FROM (a.time - b.time)) / 3600.0) AS delta_hours,
                    a.base_weight * b.base_weight AS base_score
                FROM labeled AS a
                JOIN labeled AS b
                  ON a.user_id = b.user_id
                 AND a.item_id <> b.item_id
                 AND ABS(EXTRACT(EPOCH FROM (a.time - b.time)) / 3600.0) <= ?
            ),
            covis AS (
                SELECT
                    user_id,
                    anchor_item,
                    neighbor_item,
                    SUM(base_score * EXP(-delta_hours / ?)) AS score
                FROM pairs
                GROUP BY user_id, anchor_item, neighbor_item
            ),
            ranked AS (
                SELECT
                    user_id,
                    neighbor_item AS item_id,
                    MAX(score) AS score,
                    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY MAX(score) DESC) AS rk
                FROM covis
                GROUP BY user_id, neighbor_item
            )
            SELECT user_id, item_id, score, rk
            FROM ranked
            WHERE rk <= ?
        """

        params = [lookback_start, self.cutoff, *user_id_list, self.config.window_hours, 48.0, self.config.max_per_user]
        result = con.execute(query, params).fetchall()
        for user_id, item_id, score, rank in result:
            yield Candidate(
                user_id=int(user_id),
                item_id=int(item_id),
                score=float(score),
                strategy=self.name,
                metadata={"source_rank": int(rank)},
            )

    def _build_weight_case(self) -> str:
        return "\n".join(
            f"WHEN {behavior_type} THEN {weight}" for behavior_type, weight in self.behavior_weights.items()
        )

