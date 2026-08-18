"""Integration test: windowing -> training -> evaluation -> drift detection."""
import pytest
import numpy as np
import pandas as pd
from driftx.data.windower import WindowSplitter
from driftx.training.trainer import ModelTrainer
from driftx.detection.ks_detector import KSDriftDetector
from driftx.detection.psi_detector import PSIDriftDetector


def create_drifting_temporal_data(n_rows=6000, n_features=6):
    np.random.seed(42)
    timestamps = np.arange(n_rows, dtype=float)
    
    # 3 windows of data: W0 (normal), W1 (mild drift), W2 (severe drift)
    X = np.random.normal(0, 1, (n_rows, n_features))
    # Add drift in W1 and W2
    X[2000:4000, 0] += 1.5
    X[2000:4000, 1] -= 1.0
    
    X[4000:, 0] += 3.5
    X[4000:, 1] -= 2.5
    X[4000:, 2] += 2.0
    
    # Target depends primarily on feature 0 and 1
    logit = X[:, 0] * 0.7 + X[:, 1] * 0.5
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (np.random.uniform(0, 1, n_rows) < prob).astype(int)
    
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["timestamp"] = timestamps
    df["target"] = y
    return df


class TestBaselinePipeline:

    def test_end_to_end_pipeline(self):
        df = create_drifting_temporal_data()
        
        # 1. Window dataset
        config = {
            "timestamp_col": "timestamp",
            "target_col": "target",
            "window_size": 3,
            "min_window_samples": 500,
        }
        splitter = WindowSplitter(config)
        windows = splitter.split(df)
        assert len(windows) == 3

        # 2. Train model on W0
        trainer = ModelTrainer({"type": "xgboost", "params": {"n_estimators": 50, "max_depth": 4}})
        train_res = trainer.train(windows[0].X, windows[0].y)
        assert train_res["val_accuracy"] > 0.5
        assert train_res["cost_seconds"] > 0.0

        # 3. Evaluate model across windows W0, W1, W2
        eval_w0 = trainer.evaluate(windows[0].X, windows[0].y)
        eval_w2 = trainer.evaluate(windows[2].X, windows[2].y)
        
        assert eval_w0["accuracy"] >= eval_w2["accuracy"] - 0.15 # Degradation check

        # 4. Statistical Drift Detection
        ks_detector = KSDriftDetector(alpha=0.05)
        ks_detector.set_reference(windows[0].X)
        ks_w2 = ks_detector.detect(windows[2].X)
        assert ks_w2["drift_detected"] is True

        psi_detector = PSIDriftDetector(threshold=0.2)
        psi_detector.set_reference(windows[0].X)
        psi_w2 = psi_detector.detect(windows[2].X)
        assert psi_w2["drift_detected"] is True

    def test_random_forest_trainer(self):
        df = create_drifting_temporal_data(n_rows=1000)
        trainer = ModelTrainer({"type": "random_forest", "params": {"n_estimators": 20, "max_depth": 3}})
        res = trainer.train(df[["f0", "f1", "f2"]], df["target"])
        assert res["val_accuracy"] > 0.4
        preds = trainer.predict(df[["f0", "f1", "f2"]])
        assert len(preds) == len(df)

    def test_gradient_boosting_trainer(self):
        df = create_drifting_temporal_data(n_rows=1000)
        trainer = ModelTrainer({"type": "gradient_boosting", "params": {"n_estimators": 20, "max_depth": 3}})
        res = trainer.train(df[["f0", "f1", "f2"]], df["target"])
        assert res["val_accuracy"] > 0.4
        probas = trainer.predict_proba(df[["f0", "f1", "f2"]])
        assert probas.shape == (len(df), 2)
