"""数据清洗配置模块。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.io_utils import ensure_directory


@dataclass(frozen=True)
class DataCleaningConfig:
    """数据清洗任务所需的全部路径与参数。"""

    project_root: Path
    raw_data_dir: Path
    output_dir: Path
    behavior_files: List[Path]
    item_file: Path
    output_csv_behavior: Path
    output_csv_item: Path
    output_summary: Path
    output_parquet_behavior_dir: Path
    output_parquet_item_dir: Path
    chunk_size: int
    encoding: str

    @staticmethod
    def _resolve_project_root(value: Optional[str], base_dir: Optional[Path]) -> Path:
        if value:
            path = Path(value)
            if not path.is_absolute():
                if base_dir is not None:
                    path = (base_dir / path).resolve()
                else:
                    path = path.expanduser().resolve()
            else:
                path = path.expanduser().resolve()
        else:
            if base_dir is not None:
                path = base_dir.parent.resolve()
            else:
                path = Path(__file__).resolve().parents[2]
        return path

    @classmethod
    def from_dict(
        cls,
        config_dict: Optional[Dict[str, str]] = None,
        *,
        base_dir: Optional[Path] = None,
    ) -> "DataCleaningConfig":
        config_dict = config_dict or {}

        project_root = cls._resolve_project_root(config_dict.get("project_root"), base_dir)
        raw_data_dir = (project_root / config_dict.get("raw_data_dir", "data")).resolve()
        output_dir = (project_root / config_dict.get("output_dir", "processed")).resolve()

        behavior_files_config: Iterable[str] = config_dict.get("behavior_files", [])
        if not behavior_files_config:
            behavior_files_config = (
                "data/tianchi_fresh_comp_train_user_online_partA.txt",
                "data/tianchi_fresh_comp_train_user_online_partB.txt",
            )

        behavior_files = [
            (project_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
            for path in behavior_files_config
        ]

        item_file_config = config_dict.get("item_file", "data/tianchi_fresh_comp_train_item_online.txt")
        item_file = (project_root / item_file_config).resolve() if not Path(item_file_config).is_absolute() else Path(item_file_config).resolve()

        def resolve_output_path(relative_path: str) -> Path:
            path_obj = Path(relative_path)
            if path_obj.is_absolute():
                return path_obj.resolve()
            return (output_dir / path_obj).resolve()

        output_csv_behavior = resolve_output_path(config_dict.get("output_csv_behavior", "behavior_cleaned.csv"))
        output_csv_item = resolve_output_path(config_dict.get("output_csv_item", "item_cleaned.csv"))
        output_summary = resolve_output_path(config_dict.get("output_summary", "cleaning_summary.json"))
        output_parquet_behavior_dir = resolve_output_path(
            config_dict.get("output_parquet_behavior_dir", "behavior_parquet")
        )
        output_parquet_item_dir = resolve_output_path(
            config_dict.get("output_parquet_item_dir", "item_parquet")
        )

        chunk_size = int(config_dict.get("chunk_size", 1_000_000))
        encoding = config_dict.get("encoding", "utf-8")

        ensure_directory(output_dir)
        ensure_directory(output_parquet_behavior_dir)
        ensure_directory(output_parquet_item_dir)

        return cls(
            project_root=project_root,
            raw_data_dir=raw_data_dir,
            output_dir=output_dir,
            behavior_files=behavior_files,
            item_file=item_file,
            output_csv_behavior=output_csv_behavior,
            output_csv_item=output_csv_item,
            output_summary=output_summary,
            output_parquet_behavior_dir=output_parquet_behavior_dir,
            output_parquet_item_dir=output_parquet_item_dir,
            chunk_size=chunk_size,
            encoding=encoding,
        )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "DataCleaningConfig":
        import yaml

        if not config_path.exists():
            raise FileNotFoundError(f"未找到配置文件：{config_path}")

        with config_path.open("r", encoding="utf-8") as fp:
            config_dict = yaml.safe_load(fp) or {}
        base_dir = config_path.parent.resolve()
        return cls.from_dict(config_dict, base_dir=base_dir)

