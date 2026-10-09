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

from scipy import stats
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
                - noise_floor: Dict with sampling-noise mean and std for mag/rank
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

        # Estimate sampling noise floor by repeatedly splitting shap_vals into two halves
        noise_floor = self.estimate_noise_floor(shap_vals, n_splits=25, top_k=5)

        logger.info(f"SHAP computed successfully. Top 3 features: {feature_ranking[:3]}")

        return {
            "shap_values": shap_vals,
            "mean_abs_shap": mean_abs_series,
            "feature_ranking": feature_ranking,
            "feature_rank_dict": feature_rank_dict,
            "noise_floor": noise_floor,
        }

    def estimate_noise_floor(
        self,
        shap_values: np.ndarray,
        n_splits: int = 25,
        top_k: int = 5,
    ) -> Dict[str, float]:
        """
        Estimate SHAP sampling noise mean and std on the same window and model.
        Splits shap_values into two random halves n_splits times and computes
        the magnitude and rank change scores between the halves.
        """
        n_samples, n_features = shap_values.shape
        if n_samples < 20:
            return {
                "mu_mag_noise": 0.01,
                "sd_mag_noise": 0.005,
                "mu_rank_noise": 0.05,
                "sd_rank_noise": 0.05,
            }

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        k_features = min(top_k, n_features)
        top_indices = np.argsort(mean_abs)[::-1][:k_features]

        rng = np.random.RandomState(self.seed)
        mag_noises = []
        rank_noises = []
        half_n = n_samples // 2

        for _ in range(n_splits):
            perm = rng.permutation(n_samples)
            h1 = perm[:half_n]
            h2 = perm[half_n: 2 * half_n]

            p1 = np.mean(np.abs(shap_values[h1]), axis=0)
            p2 = np.mean(np.abs(shap_values[h2]), axis=0)

            # Normalized L1 magnitude difference
            p1_norm = p1 / (np.sum(p1) + 1e-10)
            p2_norm = p2 / (np.sum(p2) + 1e-10)
            mag_score = float(np.sum(np.abs(p1_norm - p2_norm)) / 2.0)
            mag_noises.append(mag_score)

            # Rank change on top_k features
            rho, _ = stats.spearmanr(p1[top_indices], p2[top_indices])
            if np.isnan(rho):
                rho = 1.0
            rank_score = float(1.0 - rho)
            rank_noises.append(rank_score)

        return {
            "mu_mag_noise": float(np.mean(mag_noises)),
            "sd_mag_noise": float(max(np.std(mag_noises, ddof=1), 1e-5)),
            "mu_rank_noise": float(np.mean(rank_noises)),
            "sd_rank_noise": float(max(np.std(rank_noises, ddof=1), 1e-4)),
        }

    def compute_loss_shap(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values of the loss (model_output='log_loss') with labels.
        This captures concept drift by attributing model errors to features.
        """
        if len(X) > self.max_samples:
            idx = X.sample(n=self.max_samples, random_state=self.seed).index
            X_sample = X.loc[idx].reset_index(drop=True)
            y_sample = y.loc[idx].reset_index(drop=True)
        else:
            X_sample = X.reset_index(drop=True)
            y_sample = y.reset_index(drop=True)

        try:
            explainer = shap.TreeExplainer(model, data=X_sample, model_output="log_loss")
            raw_shap = explainer.shap_values(X_sample, y_sample.values)
            shap_vals = np.asarray(raw_shap)
            mean_abs = np.mean(np.abs(shap_vals), axis=0)
            mean_abs_series = pd.Series(mean_abs, index=X_sample.columns, name="loss_mean_abs_shap")
            noise_floor = self.estimate_noise_floor(shap_vals, n_splits=25, top_k=5)
            return {
                "shap_values": shap_vals,
                "mean_abs_shap": mean_abs_series,
                "noise_floor": noise_floor,
            }
        except Exception as e:
            logger.warning(f"Failed to compute loss SHAP: {e}")
            return {
                "shap_values": np.zeros((len(X_sample), X.shape[1])),
                "mean_abs_shap": pd.Series(0.0, index=X.columns),
                "noise_floor": {},
            }
