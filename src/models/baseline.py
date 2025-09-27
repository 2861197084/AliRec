"""LightGBM 基线模型实现。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import lightgbm as lgb
import pandas as pd

from src.models.data import Dataset, load_dataset


@dataclass
class LightGBMConfig:
    params: Dict[str, object]
    num_boost_round: int = 200
    early_stopping_rounds: Optional[int] = 20


class LightGBMBaseline:
    def __init__(self, config: LightGBMConfig) -> None:
        self.config = config
        self.model: Optional[lgb.Booster] = None

    def fit(self, train_path: Path, valid_path: Optional[Path] = None) -> None:
        train_dataset = load_dataset(train_path)
        X_train = train_dataset.feature_matrix()
        y_train = train_dataset.labels()

        if y_train is None:
            raise ValueError("训练数据必须包含 label 列")

        lgb_train = lgb.Dataset(X_train, label=y_train)
        valid_sets = [lgb_train]
        valid_names = ["train"]

        if valid_path is not None:
            valid_dataset = load_dataset(valid_path)
            X_valid = valid_dataset.feature_matrix()
            y_valid = valid_dataset.labels()
            if y_valid is None:
                raise ValueError("验证数据必须包含 label 列")
            lgb_valid = lgb.Dataset(X_valid, label=y_valid)
            valid_sets.append(lgb_valid)
            valid_names.append("valid")

        callbacks = [lgb.log_evaluation(period=20)]
        if valid_path is not None and self.config.early_stopping_rounds is not None:
            callbacks.append(lgb.early_stopping(self.config.early_stopping_rounds, verbose=True))

        self.model = lgb.train(
            params=self.config.params,
            train_set=lgb_train,
            num_boost_round=self.config.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict(self, data_path: Path) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        dataset = load_dataset(data_path)
        X = dataset.feature_matrix()
        best_iter = getattr(self.model, "best_iteration", None)
        predictions = self.model.predict(X, num_iteration=best_iter)

        result = dataset.meta(["user_id", "item_id"])
        result["score"] = predictions
        return result

    def save_model(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("模型尚未训练")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

