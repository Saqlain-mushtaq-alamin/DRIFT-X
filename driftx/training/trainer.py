"""Model training and evaluation module with wall-clock cost tracking."""
import logging
import time
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains, tracks, and evaluates classification models across temporal windows."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Model configuration dictionary containing:
                - type: str ("xgboost", "random_forest", "gradient_boosting")
                - params: dict of model hyperparameters
        """
        self.config = config
        self.model_type = config.get("type", "xgboost")
        self.model_params = config.get("params", {}).copy()
        self.model: Optional[Union[xgb.XGBClassifier, RandomForestClassifier, GradientBoostingClassifier]] = None
        self.training_cost_seconds: float = 0.0
        self.version: int = 0

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.15,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Train a new classifier model on provided window features and target.

        Args:
            X: Feature DataFrame
            y: Target Series
            validation_split: Fraction of window data to reserve for validation
            seed: Random seed for reproducibility

        Returns:
            Dict containing train/val metrics, training time cost, and version number.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=seed, stratify=y if len(y.unique()) > 1 else None
        )

        start_time = time.perf_counter()

        if self.model_type == "xgboost":
            params = self.model_params.copy()
            params.setdefault("random_state", seed)
            params.setdefault("n_estimators", 100)
            params.setdefault("max_depth", 5)
            # Remove legacy XGBoost parameters if present
            params.pop("use_label_encoder", None)
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        elif self.model_type == "random_forest":
            rf_keys = {"n_estimators", "max_depth", "min_samples_split", "criterion"}
            params = {k: v for k, v in self.model_params.items() if k in rf_keys}
            params.setdefault("random_state", seed)
            params.setdefault("n_estimators", 100)
            self.model = RandomForestClassifier(**params)
            self.model.fit(X_train, y_train)

        elif self.model_type == "gradient_boosting":
            gb_keys = {"n_estimators", "max_depth", "learning_rate", "subsample"}
            params = {k: v for k, v in self.model_params.items() if k in gb_keys}
            params.setdefault("random_state", seed)
            params.setdefault("n_estimators", 100)
            self.model = GradientBoostingClassifier(**params)
            self.model.fit(X_train, y_train)

        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'")

        end_time = time.perf_counter()
        self.training_cost_seconds = end_time - start_time
        self.version += 1

        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        metrics = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "val_accuracy": accuracy_score(y_val, val_pred),
            "train_f1": f1_score(y_train, train_pred, average="weighted", zero_division=0),
            "val_f1": f1_score(y_val, val_pred, average="weighted", zero_division=0),
            "cost_seconds": float(self.training_cost_seconds),
            "model_version": self.version,
            "n_train_samples": len(X_train),
        }

        logger.info(
            f"Trained {self.model_type} v{self.version}: val_acc={metrics['val_accuracy']:.4f}, "
            f"val_f1={metrics['val_f1']:.4f}, wall_cost={metrics['cost_seconds']:.3f}s"
        )
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        return self.model.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        preds = self.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "f1_weighted": f1_score(y, preds, average="weighted", zero_division=0),
            "precision_weighted": precision_score(y, preds, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y, preds, average="weighted", zero_division=0),
        }

    def get_model(self):
        return self.model
