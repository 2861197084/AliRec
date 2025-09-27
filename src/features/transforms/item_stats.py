"""商品侧统计特征。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import duckdb

from src.features.loader import FeatureDataContext


@dataclass
class ItemStatsConfig:
    lookback_days: int = 7


class ItemStatsGenerator:
    def __init__(self, context: FeatureDataContext, cutoff: datetime, config: ItemStatsConfig | None = None) -> None:
        self.context = context
        self.cutoff = cutoff
        self.config = config or ItemStatsConfig()

    def generate(self, item_ids: Iterable[int]) -> "pd.DataFrame":
        item_id_list = list(item_ids)
        if not item_id_list:
            raise ValueError("item_ids 不能为空")

        lookback_start = self.cutoff - timedelta(days=self.config.lookback_days)
        placeholders = ",".join("?" for _ in item_id_list)

        con = self.context.connect()
        query = f"""
            SELECT
                item_id,
                COUNT(*) AS item_total_events_{self.config.lookback_days}d,
                SUM(CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END) AS item_purchase_events_{self.config.lookback_days}d,
                SUM(CASE WHEN behavior_type = 3 THEN 1 ELSE 0 END) AS item_cart_events_{self.config.lookback_days}d,
                SUM(CASE WHEN behavior_type = 2 THEN 1 ELSE 0 END) AS item_fav_events_{self.config.lookback_days}d,
                SUM(CASE WHEN behavior_type = 1 THEN 1 ELSE 0 END) AS item_click_events_{self.config.lookback_days}d,
                COUNT(DISTINCT user_id) AS item_distinct_users_{self.config.lookback_days}d
            FROM read_parquet('{self.context.behavior_pattern}')
            WHERE time >= ? AND time < ?
              AND item_id IN ({placeholders})
            GROUP BY item_id
        """
        params = [lookback_start, self.cutoff, *item_id_list]
        return con.execute(query, params).fetchdf()

