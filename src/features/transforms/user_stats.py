"""用户侧统计特征。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import duckdb
import pandas as pd

from src.features.loader import FeatureDataContext


@dataclass
class UserStatsConfig:
    lookback_days: int = 7


class UserStatsGenerator:
    def __init__(self, context: FeatureDataContext, cutoff: datetime, config: UserStatsConfig | None = None) -> None:
        self.context = context
        self.cutoff = cutoff
        self.config = config or UserStatsConfig()

    def generate(self, user_ids: Iterable[int]) -> "pd.DataFrame":
        user_id_list = list(user_ids)
        if not user_id_list:
            raise ValueError("user_ids 不能为空")

        lookback_start = self.cutoff - timedelta(days=self.config.lookback_days)
        placeholders = ",".join("?" for _ in user_id_list)

        con = self.context.connect()
        query = f"""
            SELECT
                user_id,
                COUNT(*) AS total_events_{self.config.lookback_days}d,
                SUM(CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END) AS purchase_events_{self.config.lookback_days}d,
                SUM(CASE WHEN behavior_type = 3 THEN 1 ELSE 0 END) AS cart_events_{self.config.lookback_days}d,
                SUM(CASE WHEN behavior_type = 2 THEN 1 ELSE 0 END) AS fav_events_{self.config.lookback_days}d,
                SUM(CASE WHEN behavior_type = 1 THEN 1 ELSE 0 END) AS click_events_{self.config.lookback_days}d,
                COUNT(DISTINCT CAST(date_trunc('day', time) AS DATE)) AS active_days_{self.config.lookback_days}d
            FROM read_parquet('{self.context.behavior_pattern}')
            WHERE time >= ? AND time < ?
              AND user_id IN ({placeholders})
            GROUP BY user_id
        """
        params = [lookback_start, self.cutoff, *user_id_list]
        return con.execute(query, params).fetchdf()

