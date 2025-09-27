"""训练/验证数据构建工作流。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.features.builder import FeatureBuilder, FeatureBuilderConfig
from src.features.labeler import LabelConfig, LabelGenerator
from src.features.loader import create_default_context
from src.recall.loader import create_default_context as create_recall_context
from src.recall.pipeline import RecallPipeline, RecallPipelineConfig
from src.recall.strategies.recent_interaction import RecentInteractionConfig, RecentInteractionStrategy


@dataclass
class DatasetConfig:
    cutoff: datetime
    label_day: datetime | None
    lookback_days: int
    max_per_user: int
    user_limit: int | None
    output_dir: Path
    include_purchases: bool = False


def run_recall(config: DatasetConfig) -> pd.DataFrame:
    recall_context = create_recall_context()
    strategy = RecentInteractionStrategy(
        context=recall_context,
        cutoff=config.cutoff,
        config=RecentInteractionConfig(lookback_days=config.lookback_days, max_candidates_per_user=config.max_per_user),
    )
    pipeline = RecallPipeline([strategy], config=RecallPipelineConfig(max_per_user=config.max_per_user))

    con = recall_context.connect()
    limit_clause = "" if config.user_limit is None else f" LIMIT {config.user_limit}"
    user_query = f"SELECT DISTINCT user_id FROM read_parquet('{recall_context.behavior_pattern}') ORDER BY user_id{limit_clause}"
    user_rows = con.execute(user_query).fetchall()
    user_ids = [int(row[0]) for row in user_rows]

    recall_records = []
    batch_size = 1000
    for batch_start in tqdm(range(0, len(user_ids), batch_size), desc="召回候选", unit="batch"):
        batch_user_ids = user_ids[batch_start : batch_start + batch_size]
        if not batch_user_ids:
            continue
        batch_df = pipeline.run(batch_user_ids)
        recall_records.append(batch_df)
    recall_df = pd.concat(recall_records, ignore_index=True) if recall_records else pd.DataFrame(columns=["user_id", "item_id", "score", "source_strategy"])
    return recall_df


def build_dataset(config: DatasetConfig) -> pd.DataFrame:
    recall_df = run_recall(config)

    feature_context = create_default_context()
    builder = FeatureBuilder(
        context=feature_context,
        cutoff=config.cutoff,
        config=FeatureBuilderConfig(
            user_lookback_days=config.lookback_days,
            item_lookback_days=config.lookback_days,
        ),
    )
    features_df = builder.build(recall_df)

    if config.label_day is not None:
        labeler = LabelGenerator(feature_context, LabelConfig(prediction_day=config.label_day))
        if config.include_purchases:
            purchases = labeler.fetch_purchases()
            if not purchases.empty:
                purchases = purchases.assign(score=1.0, source_strategy="ground_truth")
                features_df = pd.concat(
                    [features_df, purchases.merge(features_df, on=["user_id", "item_id"], how="left", suffixes=("", "_feat"))],
                    ignore_index=True,
                )
        features_df = labeler.attach_labels(features_df)

    return features_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建训练/验证数据集")
    parser.add_argument("mode", choices=["train", "val", "test"], help="构建 dataset 模式")
    parser.add_argument("cutoff", type=str, help="召回数据截断时间（YYYY-MM-DD）")
    parser.add_argument("--label-day", type=str, default=None, help="标签日期（仅 train/val 使用）")
    parser.add_argument("--lookback-days", type=int, default=7, help="召回与特征的回溯天数")
    parser.add_argument("--max-per-user", type=int, default=300, help="每用户候选数量上限")
    parser.add_argument("--user-limit", type=int, default=None, help="调试用用户上限")
    parser.add_argument("--output-dir", type=Path, default=Path("processed/datasets"), help="输出目录")
    parser.add_argument("--include-purchases", action="store_true", help="将真实购买对强制加入候选集")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cutoff_dt = datetime.fromisoformat(args.cutoff)
    label_day_dt = datetime.fromisoformat(args.label_day) if args.label_day else None

    if args.mode in {"train", "val"} and label_day_dt is None:
        raise ValueError("train/val 模式需要提供 --label-day")

    config = DatasetConfig(
        cutoff=cutoff_dt,
        label_day=label_day_dt,
        lookback_days=args.lookback_days,
        max_per_user=args.max_per_user,
        user_limit=args.user_limit,
        output_dir=args.output_dir.resolve(),
        include_purchases=args.include_purchases,
    )

    print(f"开始构建 {args.mode} 数据集，召回截止至 {cutoff_dt.date()}...")
    dataset_df = build_dataset(config)
    print(f"构建完成，共 {len(dataset_df)} 条记录")

    output_dir = config.output_dir / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / f"features_{cutoff_dt.date()}.parquet"

    dataset_df.to_parquet(features_path, index=False)

    print(f"数据集已生成：{features_path}，记录数 {len(dataset_df)}")


if __name__ == "__main__":
    main()

