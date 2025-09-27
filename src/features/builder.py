"""特征汇总与样本构建。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.loader import FeatureDataContext
from src.features.transforms.item_stats import ItemStatsConfig, ItemStatsGenerator
from src.features.transforms.user_stats import UserStatsConfig, UserStatsGenerator


@dataclass
class FeatureBuilderConfig:
    user_lookback_days: int = 7
    item_lookback_days: int = 7


class FeatureBuilder:
    def __init__(
        self,
        context: FeatureDataContext,
        cutoff: datetime,
        config: FeatureBuilderConfig | None = None,
    ) -> None:
        self.context = context
        self.cutoff = cutoff
        self.config = config or FeatureBuilderConfig()
        self.user_generator = UserStatsGenerator(
            context=context,
            cutoff=cutoff,
            config=UserStatsConfig(lookback_days=self.config.user_lookback_days),
        )
        self.item_generator = ItemStatsGenerator(
            context=context,
            cutoff=cutoff,
            config=ItemStatsConfig(lookback_days=self.config.item_lookback_days),
        )

    def build(self, recall_df: pd.DataFrame) -> pd.DataFrame:
        if recall_df.empty:
            raise ValueError("召回结果为空，无法构建特征")

        user_ids = recall_df["user_id"].unique().tolist()
        item_ids = recall_df["item_id"].unique().tolist()

        user_features = self.user_generator.generate(user_ids)
        item_features = self.item_generator.generate(item_ids)

        result_df = recall_df.merge(user_features, on="user_id", how="left")
        result_df = result_df.merge(item_features, on="item_id", how="left")
        return result_df
