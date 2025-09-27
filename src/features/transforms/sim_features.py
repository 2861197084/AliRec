"""相似度相关特征（轻量版）。"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from src.features.loader import FeatureDataContext


@dataclass
class SimilarityFeatureConfig:
    lookback_days: int = 7


class SimilarityFeatureGenerator:
    def __init__(self, context: FeatureDataContext, cutoff, config: SimilarityFeatureConfig | None = None) -> None:
        self.context = context
        self.cutoff = cutoff
        self.config = config or SimilarityFeatureConfig()

    def generate(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if pairs.empty:
            return pd.DataFrame()

        user_ids = pairs["user_id"].unique().tolist()

        con = self.context.connect()
        query = f"""
            SELECT user_id, COUNT(DISTINCT item_id) AS user_recent_item_cnt
            FROM read_parquet('{self.context.behavior_pattern}')
            WHERE time < ? AND user_id IN ({','.join('?' for _ in user_ids)})
            GROUP BY user_id
        """
        params = [self.cutoff, *user_ids]
        user_df = con.execute(query, params).fetchdf()

        # 将用户级统计展开到 (user_id, item_id) 对
        return pairs.merge(user_df, on="user_id", how="left")
