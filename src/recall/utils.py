"""召回策略通用工具。"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable


def apply_time_decay(weight: float, delta_hours: float, tau: float = 48.0) -> float:
    """对事件按时间差应用指数衰减。"""

    if tau <= 0:
        return weight
    return weight * (2.718281828459045 ** (-delta_hours / tau))


def build_behavior_weight_case(behavior_weights: dict[int, float]) -> str:
    return "\n".join(f"WHEN {behavior} THEN {weight}" for behavior, weight in behavior_weights.items())

