"""SHAP value computation for tree-based and agnostic models."""
import sys
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Safe import to handle cv2/numpy 2.x incompatibility in headless tabular environments
try:
    import cv2
except Exception:
    sys.modules['cv2'] = None

import shap

logger = logging.getLogger(__name__)


class ShapComputer:
    """Computes SHAP values and feature importance profiles for trained classifiers."""

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        """
        Args:
            config: Configuration dict with keys:
                - shap_method: str ("tree", "kernel")
                - max_shap_samples: int (subsample limit for performance)
            seed: Random seed for SHAP subsampling. Should match the
                experiment seed for proper per-seed reproducibility.
        """
        self.config = config
        self.method = config.get("shap_method", "tree")
        self.max_samples = config.get("max_shap_samples", 500)
        self.seed = seed

    def compute(self, model: Any, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute SHAP values for model predictions on feature matrix X.

        Args:
            model: Trained classifier object.
            X: Feature DataFrame.

        Returns:
            Dict containing:
                - shap_values: np.ndarray of shape (n_samples, n_features)
                - mean_abs_shap: pd.Series of mean |SHAP| per feature
                - feature_ranking: List of feature names sorted by importance
                - feature_rank_dict: Dict mapping feature name to 1-indexed rank
        """
        if len(X) > self.max_samples:
            X_sample = X.sample(n=self.max_samples, random_state=self.seed).reset_index(drop=True)
            logger.info(f"Subsampled feature matrix from {len(X)} to {self.max_samples} for SHAP computation.")
        else:
            X_sample = X.reset_index(drop=True)

        if self.method == "tree":
            explainer = shap.TreeExplainer(model)
            raw_shap = explainer.shap_values(X_sample)
        elif self.method == "kernel":
            bg_size = min(50, len(X_sample))
            background = shap.sample(X_sample, bg_size)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            raw_shap = explainer.shap_values(X_sample)
        else:
            raise ValueError(f"Unsupported SHAP method: '{self.method}'")

        # Process SHAP array output shapes (for different shap versions and multi-class/binary output)
        if isinstance(raw_shap, list):
            shap_vals = raw_shap[1] if len(raw_shap) == 2 else raw_shap[0]
        elif hasattr(raw_shap, "values"):
            shap_vals = raw_shap.values
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
        else:
            shap_vals = raw_shap

        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

        shap_vals = np.asarray(shap_vals)

        # Compute mean absolute SHAP values per feature
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        mean_abs_series = pd.Series(mean_abs, index=X_sample.columns, name="mean_abs_shap")

        sorted_series = mean_abs_series.sort_values(ascending=False)
        feature_ranking = sorted_series.index.tolist()
        feature_rank_dict = {feat: rank + 1 for rank, feat in enumerate(feature_ranking)}

        logger.info(f"SHAP computed successfully. Top 3 features: {feature_ranking[:3]}")

        return {
            "shap_values": shap_vals,
            "mean_abs_shap": mean_abs_series,
            "feature_ranking": feature_ranking,
            "feature_rank_dict": feature_rank_dict,
        }
