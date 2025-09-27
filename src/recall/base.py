"""召回策略接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List


class Candidate:
    __slots__ = ("user_id", "item_id", "score", "strategy")

    def __init__(self, user_id: int, item_id: int, score: float, strategy: str) -> None:
        self.user_id = user_id
        self.item_id = item_id
        self.score = score
        self.strategy = strategy

    def to_tuple(self) -> tuple[int, int, float, str]:
        return self.user_id, self.item_id, self.score, self.strategy


class RecallStrategy(ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate(self, user_ids: Iterable[int]) -> Iterable[Candidate]:
        """生成候选结果。"""

    def __repr__(self) -> str:  # noqa: D401
        return f"{self.__class__.__name__}(name={self.name!r})"


