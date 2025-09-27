"""用户-商品交互特征。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import duckdb
import pandas as pd

from src.features.loader import FeatureDataContext


@dataclass
class UserItemCrossConfig:
    lookback_days: int = 7
    max_events: int = 200


class UserItemCrossGenerator:
    def __init__(self, context: FeatureDataContext, cutoff: datetime, config: UserItemCrossConfig | None = None) -> None:
        self.context = context
        self.cutoff = cutoff
        self.config = config or UserItemCrossConfig()

    def generate(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if pairs.empty:
            return pd.DataFrame()

        user_ids = pairs["user_id"].unique().tolist()
        item_ids = pairs["item_id"].unique().tolist()

        lookback_start = self.cutoff - timedelta(days=self.config.lookback_days)
        con = self.context.connect()

        query = f"""
            WITH filtered AS (
                SELECT user_id, item_id, behavior_type, time
                FROM read_parquet('{self.context.behavior_pattern}')
                WHERE time >= ? AND time < ?
                  AND user_id IN ({','.join('?' for _ in user_ids)})
                  AND item_id IN ({','.join('?' for _ in item_ids)})
            ),
            ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY user_id, item_id ORDER BY time DESC) AS rn
                FROM filtered
            ),
            agg AS (
                SELECT
                    user_id,
                    item_id,
                    COUNT(*) AS ui_total_events,
                    SUM(CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END) AS ui_purchase_cnt,
                    SUM(CASE WHEN behavior_type = 3 THEN 1 ELSE 0 END) AS ui_cart_cnt,
                    SUM(CASE WHEN behavior_type = 2 THEN 1 ELSE 0 END) AS ui_fav_cnt,
                    SUM(CASE WHEN behavior_type = 1 THEN 1 ELSE 0 END) AS ui_click_cnt
                FROM filtered
                GROUP BY user_id, item_id
            ),
            last_event AS (
                SELECT user_id, item_id, behavior_type AS last_behavior_type, time AS last_time
                FROM ranked
                WHERE rn = 1
            )
            SELECT
                agg.*,
                last_event.last_behavior_type,
                last_event.last_time
            FROM agg
            LEFT JOIN last_event USING(user_id, item_id)
        """

        params = [lookback_start, self.cutoff, *user_ids, *item_ids]
        df = con.execute(query, params).fetchdf()

        if df.empty:
            return df

        df["last_time"] = pd.to_datetime(df["last_time"])
        df["hours_since_last_interaction"] = (
            self.cutoff - df["last_time"]
        ).dt.total_seconds() / 3600.0
        df.drop(columns=["last_time"], inplace=True)
        df["last_behavior_type"] = df["last_behavior_type"].astype("Int8")

        return df
