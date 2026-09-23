"""Per-signal min-max normalization for DIS component alignment.

The three DIS input signals operate on completely different scales:
  - stat_drift_score (KS mean stat):  continuous [0, 1] but usually 0.05–0.4
  - shap_magnitude_score:             very small (L1/2 normalised), typically 0.01–0.05
  - shap_rank_change_score:           small (1 − Spearman ρ), typically 0.01–0.15

Without normalisation the weighted DIS peaks at ~0.15, which is below the
default warmup_threshold=0.3, so retraining never triggers.  This module
scales each signal to [0, 1] based on its running min/max so that the fused
DIS score occupies the full [0, 1] range and is therefore comparable to any
reasonable threshold.
"""
import numpy as np
from typing import Dict, List


class SignalNormalizer:
    """Normalizes DIS input signals to [0, 1] using running min-max scaling.

    During a short warmup period the raw values are passed through unchanged
    (there is not yet enough history to compute stable min/max).  Once at
    least ``warmup_windows`` observations have been accumulated the scaler
    uses the *full* observed range so that the normalised value reflects how
    extreme the current reading is relative to everything seen so far.

    Args:
        warmup_windows: Minimum history size before normalisation activates.
    """

    def __init__(self, warmup_windows: int = 2) -> None:
        self.warmup_windows = warmup_windows
        self.history: Dict[str, List[float]] = {
            "stat": [],
            "magnitude": [],
            "rank_change": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        stat: float,
        mag: float,
        rank: float,
    ) -> Dict[str, float]:
        """Record the three raw signal values and return normalised versions.

        Args:
            stat:  Raw statistical drift score (mean KS statistic).
            mag:   Raw SHAP magnitude score.
            rank:  Raw SHAP rank-change score.

        Returns:
            Dictionary with keys ``stat``, ``magnitude``, ``rank_change``
            each normalised to [0, 1].
        """
        self.history["stat"].append(float(stat))
        self.history["magnitude"].append(float(mag))
        self.history["rank_change"].append(float(rank))

        if len(self.history["stat"]) <= self.warmup_windows:
            # Not enough history yet for stable min/max — pass raw values through.
            # During the initial warmup (only the first warmup_windows windows of the
            # entire run, since we no longer reset on retrain) the normalizer has too
            # little history to be useful.  Raw values on their natural scales are
            # passed to DIS; the warmup_threshold in the ATC is set conservatively
            # enough to handle this.
            return {"stat": float(stat), "magnitude": float(mag), "rank_change": float(rank)}

        return {
            "stat": self._normalize(self.history["stat"], stat),
            "magnitude": self._normalize(self.history["magnitude"], mag),
            "rank_change": self._normalize(self.history["rank_change"], rank),
        }

    def reset(self) -> None:
        """Clear accumulated history (call after a retrain resets the baseline)."""
        for key in self.history:
            self.history[key] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vals: List[float], current: float) -> float:
        mn, mx = min(vals), max(vals)
        if (mx - mn) < 1e-10:
            # All observed values are identical — the channel shows no relative
            # variation.  Return the neutral mid-point (0.5) so that this channel
            # contributes a stable, non-zero weight without biasing the DIS
            # toward spurious retrains (1.0) or silent suppression (0.0).
            return 0.5
        return float(np.clip((current - mn) / (mx - mn), 0.0, 1.0))
