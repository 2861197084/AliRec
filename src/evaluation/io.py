"""评估阶段的通用输入输出工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Set

import pandas as pd


def load_pairs_from_file(file_path: Path) -> Set[str]:
    """从 TSV/CSV/Parquet 加载 user-item 对。

    允许两列命名为 `user_id`, `item_id`，或未定义列名时以首两列解析。
    若文件后缀为 `.parquet`，使用 pandas 读取后组合字符串；否则按制表符
    分隔读取文本。
    """

    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"未找到文件：{file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, sep="\t", dtype="string")

    if df.empty:
        return set()

    columns = list(df.columns)
    if len(columns) < 2:
        raise ValueError(f"文件列不足：{file_path}")

    if "user_id" in df.columns and "item_id" in df.columns:
        user_series = df["user_id"].astype("string")
        item_series = df["item_id"].astype("string")
    else:
        user_series = df.iloc[:, 0].astype("string")
        item_series = df.iloc[:, 1].astype("string")

        # 处理首行可能为表头的情况
        if user_series.iloc[0].lower() == "user_id" and item_series.iloc[0].lower() == "item_id":
            user_series = user_series.iloc[1:]
            item_series = item_series.iloc[1:]

    return {f"{u}\t{i}" for u, i in zip(user_series, item_series) if pd.notna(u) and pd.notna(i)}

