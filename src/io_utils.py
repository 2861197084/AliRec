"""通用 IO 工具函数。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_directory(directory: Path) -> None:
    """确保目录存在。"""
    directory.mkdir(parents=True, exist_ok=True)


def write_dataframe_parquet(
    df: pd.DataFrame,
    file_path: Path,
    *,
    compression: str = "snappy",
    **kwargs: Dict[str, Any],
) -> None:
    """将 DataFrame 写入 Parquet 文件，默认使用 pyarrow 引擎。"""
    ensure_directory(file_path.parent)
    df.to_parquet(
        file_path,
        engine="pyarrow",
        index=False,
        compression=compression,
        **kwargs,
    )
