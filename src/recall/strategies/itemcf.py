"""基于 ItemCF 的召回策略。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Iterator

import duckdb

from src.recall.base import Candidate, RecallStrategy
from src.recall.loader import RecallDataContext


@dataclass
class ItemCFConfig:
    lookback_days: int = 14
    max_neighbors: int = 100
    max_per_user: int = 120


class ItemCFStrategy(RecallStrategy):
    def __init__(
        self,
        context: RecallDataContext,
        cutoff: datetime,
        config: ItemCFConfig | None = None,
    ) -> None:
        super().__init__(name="itemcf")
        self.context = context
        self.cutoff = cutoff
        self.config = config or ItemCFConfig()

    def generate(self, user_ids: Iterable[int]) -> Iterator[Candidate]:
        user_id_list = list(user_ids)
        if not user_id_list:
            return iter([])

        lookback_start = self.cutoff - timedelta(days=self.config.lookback_days)
        con = self.context.connect()

        query = f"""
            WITH filtered AS (
                SELECT user_id, item_id, 1.0 AS value
                FROM read_parquet('{self.context.behavior_pattern}')
                WHERE time >= ? AND time < ?
                  AND user_id IN ({','.join('?' for _ in user_id_list)})
            ),
            item_user AS (
                SELECT item_id, ARRAY_AGG(user_id) AS users
                FROM filtered
                GROUP BY item_id
            ),
            similarity AS (
                SELECT
                    a.item_id AS item_a,
                    b.item_id AS item_b,
                    COUNT(*) AS co_count
                FROM filtered a
                JOIN filtered b ON a.user_id = b.user_id AND a.item_id <> b.item_id
                GROUP BY a.item_id, b.item_id
            ),
            ranked AS (
                SELECT
                    item_a,
                    item_b,
                    co_count,
                    ROW_NUMBER() OVER (PARTITION BY item_a ORDER BY co_count DESC) AS rk
                FROM similarity
            )
            SELECT f.user_id, r.item_b AS item_id, r.co_count AS score, r.rk
            FROM filtered f
            JOIN ranked r ON f.item_id = r.item_a
            WHERE r.rk <= ?
        """

        params = [lookback_start, self.cutoff, *user_id_list, self.config.max_neighbors]
        result = con.execute(query, params).fetchall()
        for user_id, item_id, score, rank in result:
            yield Candidate(
                user_id=int(user_id),
                item_id=int(item_id),
                score=float(score),
                strategy=self.name,
                metadata={"source_rank": int(rank)},
            )

