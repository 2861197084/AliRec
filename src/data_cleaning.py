"""数据清洗主脚本。

核心职责：
- 按配置加载原始行为与商品数据；
- 执行字段清洗、类型转换、异常过滤；
- 输出结构化 CSV 与按分区的 Parquet 文件；
- 生成清洗概要，便于后续审计。

使用示例：
```
python -m src.data_cleaning
python -m src.data_cleaning --config configs/data_cleaning.yaml
```
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.configuration import DataCleaningConfig
from src.io_utils import write_dataframe_parquet


@dataclass
class CleaningStats:
    raw_rows: int = 0
    cleaned_rows: int = 0
    dropped_missing: int = 0
    dropped_invalid_time: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "raw_rows": self.raw_rows,
            "cleaned_rows": self.cleaned_rows,
            "dropped_missing": self.dropped_missing,
            "dropped_invalid_time": self.dropped_invalid_time,
        }


BEHAVIOR_COLUMNS = [
    "user_id",
    "item_id",
    "behavior_type",
    "user_geohash",
    "item_category",
    "time",
]
ITEM_COLUMNS = ["item_id", "item_geohash", "item_category"]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _clean_behavior_chunk(chunk: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    stats.raw_rows += len(chunk)
    df = chunk.copy()
    df.replace("", pd.NA, inplace=True)

    for column in ("user_id", "item_id", "behavior_type", "item_category"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_mask = df[["user_id", "item_id", "behavior_type", "item_category"]].isna().any(axis=1)
    stats.dropped_missing += int(missing_mask.sum())
    df = df.loc[~missing_mask]

    df["behavior_type"] = df["behavior_type"].astype("int8")
    df["user_id"] = df["user_id"].astype("int64")
    df["item_id"] = df["item_id"].astype("int64")
    df["item_category"] = df["item_category"].astype("int64")

    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H", errors="coerce")
    invalid_time_mask = df["time"].isna()
    stats.dropped_invalid_time += int(invalid_time_mask.sum())
    df = df.loc[~invalid_time_mask]

    df["user_geohash"] = df["user_geohash"].astype("string")

    stats.cleaned_rows += len(df)
    return df


def clean_behavior_data(config: DataCleaningConfig, stats: CleaningStats) -> None:
    logging.info("开始清洗用户行为数据 -> %s", config.output_csv_behavior)
    if config.output_csv_behavior.exists():
        config.output_csv_behavior.unlink()
        logging.info("已删除旧的行为清洗结果：%s", config.output_csv_behavior)

    header_written = False

    for file_path in config.behavior_files:
        logging.info("读取行为数据文件：%s", file_path)
        if not file_path.exists():
            logging.warning("文件不存在，跳过：%s", file_path)
            continue

        for chunk_idx, chunk in enumerate(
            pd.read_csv(
                file_path,
                sep="\t",
                names=BEHAVIOR_COLUMNS,
                dtype="string",
                chunksize=config.chunk_size,
                encoding=config.encoding,
            )
        ):
            cleaned_chunk = _clean_behavior_chunk(chunk, stats)
            if cleaned_chunk.empty:
                logging.info("文件 %s 的第 %d 个分块清洗后为空，跳过", file_path.name, chunk_idx)
                continue

            cleaned_chunk.to_csv(
                config.output_csv_behavior,
                mode="a",
                header=not header_written,
                index=False,
                encoding=config.encoding,
            )
            header_written = True

            partition_path = config.output_parquet_behavior_dir / f"part_{file_path.stem}_{chunk_idx:04d}.parquet"
            write_dataframe_parquet(cleaned_chunk, partition_path)

            logging.info(
                "已处理 %s 分块 %d：原始 %d 行 -> 清洗后 %d 行",
                file_path.name,
                chunk_idx,
                len(chunk),
                len(cleaned_chunk),
            )

    logging.info(
        "行为数据清洗完成：原始 %d 行 -> 清洗后 %d 行，缺失剔除 %d 行，时间异常剔除 %d 行",
        stats.raw_rows,
        stats.cleaned_rows,
        stats.dropped_missing,
        stats.dropped_invalid_time,
    )


def clean_item_data(config: DataCleaningConfig) -> Dict[str, int]:
    logging.info("开始清洗商品子集数据 -> %s", config.output_csv_item)
    if not config.item_file.exists():
        raise FileNotFoundError(f"未找到商品子集数据：{config.item_file}")

    df = pd.read_csv(
        config.item_file,
        sep="\t",
        names=ITEM_COLUMNS,
        dtype="string",
        encoding=config.encoding,
    )

    raw_rows = len(df)
    df.replace("", pd.NA, inplace=True)

    for column in ("item_id", "item_category"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["item_geohash"] = df["item_geohash"].astype("string")

    df = df.dropna(subset=["item_id", "item_category"])
    df["item_id"] = df["item_id"].astype("int64")
    df["item_category"] = df["item_category"].astype("int64")
    df = df.drop_duplicates(subset=["item_id", "item_geohash", "item_category"], keep="first")

    df.to_csv(config.output_csv_item, index=False, encoding=config.encoding)
    write_dataframe_parquet(df, config.output_parquet_item_dir / "items.parquet")

    cleaned_rows = len(df)
    logging.info("商品子集清洗完成：原始 %d 行 -> 清洗后 %d 行", raw_rows, cleaned_rows)
    return {"raw_rows": raw_rows, "cleaned_rows": int(cleaned_rows)}


def write_summary(summary_path: Path, behavior_stats: CleaningStats, item_stats: Dict[str, int]) -> None:
    summary = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "behavior_data": behavior_stats.to_dict(),
        "item_data": item_stats,
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    logging.info("清洗摘要已写入：%s", summary_path)


def load_config(config_path: Optional[str]) -> DataCleaningConfig:
    if config_path is None:
        logging.info("未提供配置文件，使用默认配置")
        return DataCleaningConfig.from_dict()

    logging.info("已加载配置文件：%s", config_path)
    return DataCleaningConfig.from_yaml(Path(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗天池移动推荐数据")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="指定 YAML 配置文件路径",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)

    behavior_stats = CleaningStats()
    clean_behavior_data(config, behavior_stats)
    item_stats = clean_item_data(config)
    write_summary(config.output_summary, behavior_stats, item_stats)


if __name__ == "__main__":
    main()
