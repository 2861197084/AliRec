"""标签生成工具。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import duckdb
import pandas as pd

from src.features.loader import FeatureDataContext


@dataclass
class LabelConfig:
    prediction_day: datetime


class LabelGenerator:
    def __init__(self, context: FeatureDataContext, config: LabelConfig) -> None:
        self.context = context
        self.config = config

    def fetch_purchases(self) -> pd.DataFrame:
        day_start = datetime(self.config.prediction_day.year, self.config.prediction_day.month, self.config.prediction_day.day)
        day_end = day_start + pd.Timedelta(days=1)

        con = self.context.connect()
        query = f"""
            SELECT user_id, item_id
            FROM read_parquet('{self.context.behavior_pattern}')
            WHERE behavior_type = 4
              AND time >= ?
              AND time < ?
        """
        return con.execute(query, [day_start, day_end]).fetchdf()

    def attach_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        purchases = self.fetch_purchases()
        purchases["label"] = 1

        labeled_df = features_df.merge(purchases, on=["user_id", "item_id"], how="left")
        labeled_df["label"] = labeled_df["label"].fillna(0).astype("int8")
        return labeled_df

