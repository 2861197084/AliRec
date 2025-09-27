"""命令行评估入口。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.evaluation.io import load_pairs_from_file
from src.evaluation.metrics import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估预测结果的精确率/召回率/F1")
    parser.add_argument("prediction", type=Path, help="预测结果文件路径")
    parser.add_argument("truth", type=Path, help="真实购买集合文件路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选，指定将评估结果以 JSON 写出的路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_pairs = load_pairs_from_file(args.prediction)
    truth_pairs = load_pairs_from_file(args.truth)
    result = evaluate(prediction_pairs, truth_pairs)

    result_dict = asdict(result)
    print(json.dumps(result_dict, ensure_ascii=False, indent=2))

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(result_dict, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

