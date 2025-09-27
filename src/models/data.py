"""训练数据加载工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass
class Dataset:
    data: pd.DataFrame
    label_column: str = "label"
    meta_columns: Sequence[str] = ("user_id", "item_id", "source_strategy", "source_rank", "source_count")

    def feature_matrix(self) -> pd.DataFrame:
        drop_cols = [col for col in self.meta_columns if col in self.data.columns]
        if self.label_column in self.data.columns:
            drop_cols.append(self.label_column)
        drop_cols = list(dict.fromkeys(drop_cols))
        return self.data.drop(columns=drop_cols)

    def labels(self) -> pd.Series | None:
        if self.label_column not in self.data.columns:
            return None
        return self.data[self.label_column]

    def meta(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        selected = columns or self.meta_columns
        available = [column for column in selected if column in self.data.columns]
        return self.data[available].copy()

    def truth_pairs(self) -> pd.DataFrame:
        if self.label_column not in self.data.columns:
            raise ValueError("标签列不存在")
        return self.data.loc[self.data[self.label_column] == 1, ["user_id", "item_id"]]


def load_dataset(path: Path) -> Dataset:
    df = pd.read_parquet(path)
    return Dataset(data=df)

