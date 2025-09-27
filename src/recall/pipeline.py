"""召回策略组合与结果合并。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict

import pandas as pd

from src.recall.base import RecallStrategy


@dataclass
class RecallPipelineConfig:
    max_per_user: int = 300


class RecallPipeline:
    def __init__(
        self,
        strategies: List[RecallStrategy],
        config: RecallPipelineConfig | None = None,
        item_whitelist: frozenset[int] | None = None,
    ) -> None:
        self.strategies = strategies
        self.config = config or RecallPipelineConfig()
        self.item_whitelist = item_whitelist

    def run(self, user_ids: Iterable[int]) -> pd.DataFrame:
        aggregated: dict[int, dict[int, dict[str, object]]] = {}
        strategy_counts: Dict[str, int] = {strategy.name: 0 for strategy in self.strategies}

        for strategy in self.strategies:
            for candidate in strategy.generate(user_ids):
                if self.item_whitelist is not None and candidate.item_id not in self.item_whitelist:
                    continue
                strategy_counts[strategy.name] += 1

                user_dict = aggregated.setdefault(candidate.user_id, {})
                existing = user_dict.get(candidate.item_id)
                if existing is None or existing["score"] < candidate.score:
                    metadata = dict(candidate.metadata)
                    metadata.setdefault("source_strategies", {candidate.strategy})
                    user_dict[candidate.item_id] = {
                        "score": candidate.score,
                        "strategy": candidate.strategy,
                        "metadata": metadata,
                    }
                else:
                    existing_metadata = existing["metadata"]
                    existing_metadata.setdefault("source_strategies", set()).add(candidate.strategy)
                    if "source_rank" in candidate.metadata:
                        existing_metadata["source_rank"] = min(
                            existing_metadata.get("source_rank", candidate.metadata["source_rank"]),
                            candidate.metadata["source_rank"],
                        )

        rows = []
        for user_id, item_map in aggregated.items():
            sorted_items = sorted(item_map.items(), key=lambda item: item[1]["score"], reverse=True)
            limited_items = sorted_items[: self.config.max_per_user]
            for item_id, info in limited_items:
                metadata = info["metadata"]
                rows.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "score": info["score"],
                        "source_strategy": info["strategy"],
                        "source_rank": metadata.get("source_rank"),
                        "source_count": len(metadata.get("source_strategies", {info["strategy"]})),
                    }
                )

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            total_candidates = len(result_df)
            coverage_info = ", ".join(
                f"{name}: {count}" for name, count in strategy_counts.items()
            )
            print(f"召回合并完成，共 {total_candidates} 条候选，各策略贡献：{coverage_info}")
        else:
            print("召回结果为空，请检查策略配置")

        return result_df

