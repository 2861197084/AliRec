"""特征生成命令行工具。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.features.builder import FeatureBuilder, FeatureBuilderConfig
from src.features.labeler import LabelConfig, LabelGenerator
from src.features.loader import create_default_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="依据召回结果构建特征表")
    parser.add_argument("cutoff", type=str, help="预测日（YYYY-MM-DD），特征窗口在此之前")
    parser.add_argument("recall", type=Path, help="召回候选 Parquet 文件路径")
    parser.add_argument("--output", type=Path, default=Path("processed/features/features.parquet"), help="特征输出路径")
    parser.add_argument("--user-lookback", type=int, default=7, help="用户特征回溯天数")
    parser.add_argument("--item-lookback", type=int, default=7, help="商品特征回溯天数")
    parser.add_argument("--label-day", type=str, default=None, help="可选，指定打标签的日期（YYYY-MM-DD）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoff = datetime.fromisoformat(args.cutoff)

    recall_path = args.recall.resolve()
    recall_df = pd.read_parquet(recall_path)

    context = create_default_context()
    builder = FeatureBuilder(
        context=context,
        cutoff=cutoff,
        config=FeatureBuilderConfig(
            user_lookback_days=args.user_lookback,
            item_lookback_days=args.item_lookback,
        ),
    )
    features_df = builder.build(recall_df)

    if args.label_day is not None:
        label_day = datetime.fromisoformat(args.label_day)
        labeler = LabelGenerator(
            context=context,
            config=LabelConfig(prediction_day=label_day),
        )
        features_df = labeler.attach_labels(features_df)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    suffix = "(含标签)" if args.label_day is not None else ""
    print(f"特征已输出至：{output_path}，共 {len(features_df)} 条候选 {suffix}。")


if __name__ == "__main__":
    main()
