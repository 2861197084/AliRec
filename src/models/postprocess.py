"""预测后处理与提交文件生成。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SubmissionConfig:
    strategy: str
    parameter: float | int
    per_user_cap: int


def select_predictions(scores: pd.DataFrame, config: SubmissionConfig) -> pd.DataFrame:
    if config.strategy == "threshold":
        selected = (
            scores[scores["score"] >= float(config.parameter)]
            .sort_values(["user_id", "score"], ascending=[True, False])
            .groupby("user_id")
            .head(config.per_user_cap)
        )
    else:
        topk = int(config.parameter)
        selected = (
            scores.sort_values("score", ascending=False)
            .groupby("user_id")
            .head(min(topk, config.per_user_cap))
        )
    return selected


def write_submission(selected: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected[["user_id", "item_id"]].to_csv(path, sep="\t", index=False, header=False)
