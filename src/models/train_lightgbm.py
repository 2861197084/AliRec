"""LightGBM 基线训练脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import evaluate
from src.models.baseline import LightGBMBaseline, LightGBMConfig


DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 LightGBM 基线模型")
    parser.add_argument("train", type=Path, help="训练数据 Parquet 路径")
    parser.add_argument("--valid", type=Path, default=None, help="可选，验证数据 Parquet 路径")
    parser.add_argument("--output-model", type=Path, default=Path("processed/models/lightgbm.txt"), help="模型输出路径")
    parser.add_argument("--prediction-output", type=Path, default=Path("processed/predictions/lightgbm_valid.parquet"), help="验证集预测输出路径")
    parser.add_argument("--top-k", type=int, default=None, help="评估时选取的 Top-K 预测数量，默认与真实正例数量一致")
    parser.add_argument("--predict", type=Path, default=None, help="可选，指定生成预测得分的数据集路径（无标签）")
    parser.add_argument("--submission-output", type=Path, default=None, help="若指定，将预测结果按照 F1 提交格式输出 TSV")
    parser.add_argument("--submission-size", type=int, default=50000, help="提交文件中保留的 Top-N 结果")
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--early-stopping", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = LightGBMConfig(
        params=DEFAULT_PARAMS,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping,
    )
    model = LightGBMBaseline(config)
    print("开始训练 LightGBM 模型...")
    model.fit(args.train, args.valid)
    print("训练完成")

    model.save_model(args.output_model)

    if args.valid is not None:
        print("生成验证集预测...")
        pred_df = model.predict(args.valid)
        pred_output_path = args.prediction_output.resolve()
        pred_output_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(pred_output_path, index=False)

        valid_df = pd.read_parquet(args.valid)
        truth_rows = valid_df.loc[valid_df["label"] == 1, ["user_id", "item_id"]]
        truth_pairs = {f"{row.user_id}\t{row.item_id}" for _, row in truth_rows.iterrows()}

        if truth_pairs:
            top_k = args.top_k or len(truth_pairs)
            prediction_pairs = set(
                pred_df.sort_values("score", ascending=False)
                .head(top_k)
                [["user_id", "item_id"]]
                .apply(lambda r: f"{r.user_id}\t{r.item_id}", axis=1)
            )
            metrics = evaluate(prediction_pairs, truth_pairs)
            print(f"验证集指标：Precision={metrics.precision:.4f}, Recall={metrics.recall:.4f}, F1={metrics.f1:.4f}")
        else:
            print("验证集中无正样本(label=1)，请扩大召回范围后重试评估。")

    if args.predict is not None:
        print("生成预测数据...")
        predict_df = model.predict(args.predict)
        predict_output_path = args.submission_output.resolve() if args.submission_output else None
        if predict_output_path is not None:
            predict_output_path.parent.mkdir(parents=True, exist_ok=True)
            if predict_output_path.suffix.lower() in {".tsv", ""}:
                submission_size = args.submission_size
                submission_df = predict_df.sort_values("score", ascending=False).head(submission_size)
                submission_df[["user_id", "item_id"]].to_csv(
                    predict_output_path,
                    sep="\t",
                    header=False,
                    index=False,
                )
            else:
                predict_df.to_parquet(predict_output_path, index=False)
        else:
            print(predict_df.head())


if __name__ == "__main__":
    main()
