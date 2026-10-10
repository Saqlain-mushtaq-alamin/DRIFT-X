"""P14: Oracle retraining policy (retrains exactly at ground-truth change points)."""
from typing import Dict, Any, Iterable, Tuple
import pandas as pd
import numpy as np
from driftx.policy.base import RetrainingPolicy, PolicyDecision


class OraclePolicy(RetrainingPolicy):
    """Oracle policy that retrains precisely when a true change point occurs."""

    def __init__(self, change_points: Iterable[int] = ()):
        self.cps = {int(c) for c in change_points}

    @property
    def name(self) -> str:
        return "Oracle (true change points)"

    @property
    def policy_id(self) -> str:
        return "p14_oracle"

    @property
    def needs_shap(self) -> bool:
        return False

    def decide(
        self,
        window_id: int,
        stat_drift: Dict[str, Any],
        shap_magnitude: Dict[str, Any],
        shap_rank_change: Dict[str, Any],
        dis_result: Dict[str, Any],
        atc_result: Dict[str, Any],
        **kwargs: Any,
    ) -> PolicyDecision:
        window = kwargs.get("window")
        is_cp = window_id in self.cps
        if not is_cp and window is not None and hasattr(window, "data") and "true_cp" in window.data.columns:
            is_cp = bool(window.data["true_cp"].iloc[0] == 1)

        return PolicyDecision(
            should_retrain=is_cp,
            reason=f"P14: CP={is_cp}",
            policy_name=self.name,
            window_id=window_id,
            evidence={"is_changepoint": is_cp},
        )


def cp_scores(res_pol: pd.DataFrame, cps: Iterable[int], tol: int = 2) -> Tuple[float, float]:
    """Compute recall and precision of retrain events against true change points."""
    r = set(res_pol[(res_pol["window_id"] > 0) & res_pol["retrained"]]["window_id"])
    cps_set = set(cps)
    if not cps_set:
        return 0.0, float("nan")
    hit = {c for c in cps_set if any(c <= w <= c + tol for w in r)}
    good = {w for w in r if any(c <= w <= c + tol for c in cps_set)}
    recall = len(hit) / len(cps_set)
    precision = len(good) / len(r) if r else float("nan")
    return recall, precision
