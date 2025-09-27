from pathlib import Path

import pandas as pd

from src.evaluation.io import load_pairs_from_file
from src.evaluation.metrics import compute_f1, compute_precision, compute_recall, evaluate


def test_metrics_basic_case() -> None:
    prediction = {"1\t1", "1\t2", "2\t1"}
    truth = {"1\t1", "2\t1"}

    precision = compute_precision(prediction, truth)
    recall = compute_recall(prediction, truth)
    f1 = compute_f1(prediction, truth)
    result = evaluate(prediction, truth)

    assert abs(precision - 2 / 3) < 1e-9
    assert abs(recall - 1.0) < 1e-9
    assert abs(f1 - (2 * (2 / 3) * 1.0) / ((2 / 3) + 1.0)) < 1e-9
    assert abs(result.precision - precision) < 1e-9
    assert abs(result.recall - recall) < 1e-9
    assert abs(result.f1 - f1) < 1e-9


def test_load_pairs_from_csv(tmp_path: Path) -> None:
    file_path = tmp_path / "pred.tsv"
    file_path.write_text("user_id\titem_id\n1\t2\n3\t4\n", encoding="utf-8")
    pairs = load_pairs_from_file(file_path)
    assert pairs == {"1\t2", "3\t4"}


def test_load_pairs_from_parquet(tmp_path: Path) -> None:
    file_path = tmp_path / "truth.parquet"
    df = pd.DataFrame({"user_id": ["1", "2"], "item_id": ["3", "4"]})
    df.to_parquet(file_path, index=False)

    pairs = load_pairs_from_file(file_path)
    assert pairs == {"1\t3", "2\t4"}

