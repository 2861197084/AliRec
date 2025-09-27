"""LightGBM 基线训练脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.f1_search import search_best_f1
from src.evaluation.metrics import evaluate
from src.models.baseline import LightGBMBaseline, LightGBMConfig
from src.models.data import load_dataset
from src.models.postprocess import SubmissionConfig, select_predictions, write_submission


DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 128,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.01,
    "lambda_l2": 1.0,
    "verbose": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 LightGBM 基线模型")
    parser.add_argument("train", type=Path, help="训练数据 Parquet 路径")
    parser.add_argument("--valid", type=Path, default=None, help="可选，验证数据 Parquet 路径")
    parser.add_argument("--output-model", type=Path, default=Path("processed/models/lightgbm.txt"), help="模型输出路径")
    parser.add_argument("--prediction-output", type=Path, default=Path("processed/predictions/lightgbm_valid.parquet"), help="验证集预测输出路径")
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--early-stopping", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="LightGBM 设备类型")
    parser.add_argument("--f1-topk", type=int, nargs="*", default=list(range(1, 31)))
    parser.add_argument("--f1-threshold", type=float, nargs="*", default=[i / 100 for i in range(5, 96, 5)])
    parser.add_argument("--per-user-cap", type=int, default=20)
    parser.add_argument("--predict", type=Path, default=None, help="可选，指定生成预测得分的数据集路径")
    parser.add_argument("--submission-output", type=Path, default=None, help="若指定，将输出提交文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    params = DEFAULT_PARAMS.copy()
    if args.device == "gpu":
        params.update({"device_type": "gpu"})

    config = LightGBMConfig(
        params=params,
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

        valid_dataset = load_dataset(args.valid)
        truth_df = valid_dataset.truth_pairs()

        if truth_df.empty:
            print("验证集中无正样本(label=1)，无法进行 F1 搜索。")
        else:
            best_result = search_best_f1(
                pred_df,
                truth_df,
                threshold_grid=args.f1_threshold,
                topk_grid=args.f1_topk,
                per_user_cap=args.per_user_cap,
            )
            print(
                f"验证集最优 F1：{best_result.f1:.4f} (策略={best_result.strategy}, 参数={best_result.parameter}, cap={best_result.per_user_cap})"
            )

            if args.predict is not None:
                print("生成预测数据...")
                predict_df = model.predict(args.predict)
                submission_config = SubmissionConfig(
                    strategy=best_result.strategy,
                    parameter=best_result.parameter,
                    per_user_cap=best_result.per_user_cap,
                )
                selected = select_predictions(predict_df, submission_config)

                if args.submission_output is not None:
                    write_submission(selected, args.submission_output.resolve())
                    print(f"提交文件已生成：{args.submission_output}")


if __name__ == "__main__":
    main()
