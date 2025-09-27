"""F1 网格搜索工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate


@dataclass
class F1SearchResult:
    strategy: str
    parameter: float | int
    per_user_cap: int
    precision: float
    recall: float
    f1: float


def _build_prediction_pairs(df: pd.DataFrame) -> set[str]:
    return set(df.apply(lambda r: f"{r.user_id}\t{r.item_id}", axis=1))


def f1_search_threshold(
    scores: pd.DataFrame,
    truth_pairs: set[str],
    thresholds: Iterable[float],
    per_user_cap: int,
) -> F1SearchResult:
    best = F1SearchResult("threshold", 0.0, per_user_cap, 0.0, 0.0, 0.0)

    for threshold in thresholds:
        selected = (
            scores[scores["score"] >= threshold]
            .sort_values(["user_id", "score"], ascending=[True, False])
            .groupby("user_id")
            .head(per_user_cap)
        )
        prediction_pairs = _build_prediction_pairs(selected)
        metrics = evaluate(prediction_pairs, truth_pairs)
        if metrics.f1 > best.f1:
            best = F1SearchResult("threshold", threshold, per_user_cap, metrics.precision, metrics.recall, metrics.f1)

    return best


def f1_search_topk(
    scores: pd.DataFrame,
    truth_pairs: set[str],
    topks: Iterable[int],
    per_user_cap: int,
) -> F1SearchResult:
    best = F1SearchResult("topk", 0, per_user_cap, 0.0, 0.0, 0.0)

    for topk in topks:
        selected = (
            scores.sort_values("score", ascending=False)
            .groupby("user_id")
            .head(min(topk, per_user_cap))
        )
        prediction_pairs = _build_prediction_pairs(selected)
        metrics = evaluate(prediction_pairs, truth_pairs)
        if metrics.f1 > best.f1:
            best = F1SearchResult("topk", topk, per_user_cap, metrics.precision, metrics.recall, metrics.f1)

    return best


def search_best_f1(
    scores: pd.DataFrame,
    truth_df: pd.DataFrame,
    threshold_grid: Iterable[float],
    topk_grid: Iterable[int],
    per_user_cap: int,
) -> F1SearchResult:
    truth_pairs = _build_prediction_pairs(truth_df)

    best_threshold = f1_search_threshold(scores, truth_pairs, threshold_grid, per_user_cap)
    best_topk = f1_search_topk(scores, truth_pairs, topk_grid, per_user_cap)

    return best_threshold if best_threshold.f1 >= best_topk.f1 else best_topk
