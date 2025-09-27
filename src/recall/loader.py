"""召回阶段的数据加载封装。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb


@dataclass(frozen=True)
class RecallDataContext:
    """召回阶段共享的数据上下文。"""

    behavior_pattern: str
    item_path: Path

    def connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(database=":memory:")
        con.execute("PRAGMA threads=4")
        return con


def _ensure_exists(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description}不存在: {resolved}")
    return resolved


def create_default_context(project_root: Optional[Path] = None) -> RecallDataContext:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    root = project_root.resolve()

    behavior_dir = _ensure_exists(root / "processed" / "behavior_parquet", "行为数据目录")
    item_path = _ensure_exists(root / "processed" / "item_parquet" / "items.parquet", "商品子集 Parquet")
    behavior_pattern = (behavior_dir / "*.parquet").as_posix()
    return RecallDataContext(behavior_pattern=behavior_pattern, item_path=item_path)

