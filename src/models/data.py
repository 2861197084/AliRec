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
    meta_columns: Sequence[str] = ("user_id", "item_id", "source_strategy")

    def feature_matrix(self) -> pd.DataFrame:
        drop_cols = list(self.meta_columns)
        drop_cols = [column for column in drop_cols if column in self.data.columns]
        if self.label_column in self.data.columns:
            drop_cols.append(self.label_column)
        return self.data.drop(columns=drop_cols)

    def labels(self) -> pd.Series | None:
        if self.label_column not in self.data.columns:
            return None
        return self.data[self.label_column]

    def meta(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        selected = columns or self.meta_columns
        available = [column for column in selected if column in self.data.columns]
        return self.data[available].copy()


def load_dataset(path: Path) -> Dataset:
    df = pd.read_parquet(path)
    return Dataset(data=df)

