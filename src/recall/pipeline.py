"""召回策略组合与结果合并。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.recall.base import Candidate, RecallStrategy


@dataclass
class RecallPipelineConfig:
    max_per_user: int = 300


class RecallPipeline:
    def __init__(self, strategies: List[RecallStrategy], config: RecallPipelineConfig | None = None) -> None:
        self.strategies = strategies
        self.config = config or RecallPipelineConfig()

    def run(self, user_ids: Iterable[int]) -> pd.DataFrame:
        aggregated: dict[int, dict[int, tuple[float, str]]] = {}

        for strategy in self.strategies:
            for candidate in strategy.generate(user_ids):
                user_dict = aggregated.setdefault(candidate.user_id, {})
                existing = user_dict.get(candidate.item_id)
                if existing is None or existing[0] < candidate.score:
                    user_dict[candidate.item_id] = (candidate.score, candidate.strategy)

        rows = []
        for user_id, item_map in aggregated.items():
            sorted_items = sorted(item_map.items(), key=lambda item: item[1][0], reverse=True)
            limited_items = sorted_items[: self.config.max_per_user]
            for item_id, (score, strategy) in limited_items:
                rows.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "score": score,
                        "source_strategy": strategy,
                    }
                )

        return pd.DataFrame(rows)

