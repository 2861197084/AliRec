"""评估指标计算工具。

提供精确率、召回率与 F1 的计算函数，支持批量预测结果
或单个预测集合的评估。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set


@dataclass(frozen=True)
class EvaluationResult:
    """封装离线预测评估指标。"""

    precision: float
    recall: float
    f1: float


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_precision(prediction: Set[str], truth: Set[str]) -> float:
    """计算精确率。

    参数使用字符串集合，默认形如 "user_id\titem_id"，以便与输出格式对齐。
    """

    if not prediction:
        return 0.0
    hits = len(prediction & truth)
    return hits / len(prediction)


def compute_recall(prediction: Set[str], truth: Set[str]) -> float:
    """计算召回率。"""

    if not truth:
        return 0.0
    hits = len(prediction & truth)
    return hits / len(truth)


def compute_f1(prediction: Set[str], truth: Set[str]) -> float:
    """计算 F1 值。"""

    precision = compute_precision(prediction, truth)
    recall = compute_recall(prediction, truth)
    return _safe_divide(2 * precision * recall, precision + recall)


def evaluate(prediction: Set[str], truth: Set[str]) -> EvaluationResult:
    """综合计算精确率、召回率与 F1。"""

    precision = compute_precision(prediction, truth)
    recall = compute_recall(prediction, truth)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return EvaluationResult(precision=precision, recall=recall, f1=f1)

