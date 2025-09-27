"""基于近期行为的召回策略。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Iterator

import duckdb

from src.recall.base import Candidate, RecallStrategy
from src.recall.loader import RecallDataContext


@dataclass
class RecentInteractionConfig:
    lookback_days: int = 7
    max_candidates_per_user: int = 200
    behavior_weights: dict[int, float] | None = None


class RecentInteractionStrategy(RecallStrategy):
    def __init__(
        self,
        context: RecallDataContext,
        cutoff: datetime,
        config: RecentInteractionConfig | None = None,
    ) -> None:
        super().__init__(name="recent_interaction")
        self.context = context
        self.cutoff = cutoff
        self.config = config or RecentInteractionConfig()
        self.behavior_weights = self.config.behavior_weights or {
            4: 1.0,
            3: 0.7,
            2: 0.5,
            1: 0.1,
        }

    def generate(self, user_ids: Iterable[int]) -> Iterator[Candidate]:
        user_id_list = list(user_ids)
        if not user_id_list:
            return iter([])

        lookback_start = self.cutoff - timedelta(days=self.config.lookback_days)
        con = self.context.connect()

        query = f"""
            WITH recent_behavior AS (
                SELECT
                    user_id,
                    item_id,
                    behavior_type,
                    time
                FROM read_parquet('{self.context.behavior_pattern}')
                WHERE time >= ? AND time < ?
                    AND user_id IN ({','.join('?' for _ in user_id_list)})
            )
            SELECT * FROM (
                SELECT
                    rb.user_id,
                    rb.item_id,
                    SUM(weight) AS score,
                    ROW_NUMBER() OVER (PARTITION BY rb.user_id ORDER BY SUM(weight) DESC) AS rk
                FROM (
                    SELECT
                        user_id,
                        item_id,
                        behavior_type,
                        time,
                        CASE behavior_type
                            {self._build_weight_case()}
                            ELSE 0.0
                        END AS weight
                    FROM recent_behavior
                ) AS rb
                GROUP BY rb.user_id, rb.item_id
            )
            WHERE rk <= ?
        """

        params = [lookback_start, self.cutoff, *user_id_list, self.config.max_candidates_per_user]
        result = con.execute(query, params).fetchall()
        for user_id, item_id, score, _rank in result:
            yield Candidate(user_id=int(user_id), item_id=int(item_id), score=float(score), strategy=self.name)

    def _build_weight_case(self) -> str:
        return "\n".join(
            f"WHEN {behavior_type} THEN {weight}" for behavior_type, weight in self.behavior_weights.items()
        )

