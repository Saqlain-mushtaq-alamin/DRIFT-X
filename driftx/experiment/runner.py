"""
Main experiment orchestrator — runs the full DRIFT-X pipeline.

Evaluation protocol: **test-then-train** (prequential evaluation).

For each policy × each seed:
  1. Load data and create windows
  2. Train initial model on window 0 (no accuracy logged — no prior model)
  3. For each subsequent window w (w = 1..N):
     a. EVALUATE current model on window w  ← out-of-sample (model has NOT seen w)
     b. Compute statistical drift on window w
     c. Compute SHAP values on window w
     d. Compute DIS (fused score)
     e. Ask the active policy: retrain?
     f. If yes: retrain on window w, log cost
        — the logged accuracy is the PRE-RETRAIN eval (honest, out-of-sample)
     g. Log everything to MLflow + CSV

This protocol is the only way to get an honest comparison across policies with
different retrain frequencies.  Policies that retrain more often are no longer
artificially rewarded with in-sample accuracy inflation.
"""
import copy
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

try:
    import mlflow
except ImportError:
    mlflow = None

from driftx.data.ingestor import DataIngestor
from driftx.data.windower import WindowSplitter
from driftx.detection.ks_detector import KSDriftDetector
from driftx.detection.psi_detector import PSIDriftDetector
from driftx.explainability.shap_computer import ShapComputer
from driftx.explainability.magnitude import MagnitudeTracker
from driftx.explainability.rank_change import RankChangeTracker
from driftx.fusion.dis import DriftImpactScore
from driftx.fusion.normalizer import SignalNormalizer
from driftx.fusion.threshold import AdaptiveThresholdController
from driftx.training.trainer import ModelTrainer
from driftx.policy import create_all_policies
from driftx.policy.base import PolicyDecision

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the full DRIFT-X experiment pipeline."""
    
    def __init__(self, config: dict):
        self.config = config
        self.results: List[Dict] = []
    
    def run_all(self, df_override: Optional[pd.DataFrame] = None, output_filename: Optional[str] = None) -> pd.DataFrame:
        """
        Run all policies × all seeds. Main entry point.
        
        Args:
            df_override: Optional pre-loaded DataFrame (e.g. for testing)
            output_filename: Optional override for the output CSV filename.
                Defaults to 'experiment_results.csv'.
            
        Returns:
            DataFrame with one row per (policy, seed, window).
        """
        seeds = self._get_seeds()
        policies = create_all_policies(self.config["policy"])
        
        if df_override is not None:
            df = df_override
        else:
            ingestor = DataIngestor(self.config["data"])
            data_dir = self.config["data"].get("data_dir", None)
            if not data_dir:
                project_root = Path(__file__).resolve().parent.parent.parent
                data_dir = str(project_root / "data" / "raw")
            
            data_path = Path(data_dir)
            data_path.mkdir(parents=True, exist_ok=True)
            
            # If dataset file doesn't exist, auto-generate synthetic drift data
            raw_files = list(data_path.glob("*.csv")) + list(data_path.glob("*.parquet"))
            if not raw_files:
                from scripts.download_data import generate_synthetic_drift_data
                synth_file = data_path / "synthetic_drift.csv"
                generate_synthetic_drift_data(
                    output_path=str(synth_file),
                    n_windows=20,
                    samples_per_window=1500,
                    n_features=15,
                    seed=42,
                    drift_intensity=0.4,
                )
            
            df = ingestor.load(data_dir=str(data_path))
        
        splitter = WindowSplitter(self.config["data"])
        windows = splitter.split(df)
        
        logger.info(
            f"Running {len(policies)} policies × "
            f"{len(seeds)} seeds × {len(windows)} windows"
        )
        
        # Setup MLflow Tracking URI and Experiment Name if available
        if mlflow is not None:
            mlflow_cfg = self.config.get("mlflow", {})
            tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
            exp_name = mlflow_cfg.get("experiment_name", "driftx_main")
            try:
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(exp_name)
            except Exception as e:
                logger.warning(f"MLflow setup warning: {e}")
        
        all_results = []
        timestamp_col = self.config.get("data", {}).get("timestamp_col", "TransactionDT")
        
        for policy_id, policy in policies.items():
            for seed in seeds:
                logger.info(f"\n{'='*60}")
                logger.info(f"Policy: {policy.name} | Seed: {seed}")
                logger.info(f"{'='*60}")

                # Apply per-seed data replicate shift if configured.
                # When data_replicate_shift_windows > 0, each seed sees a
                # different temporal slice of the data (distinct train/eval
                # windows), not just a different model initialisation seed.
                # This is a stronger variance estimate than model-seed-only.
                has_replicate = (
                    self.config.get("project", {}).get("data_replicate_offset_rows", 0) > 0
                    or self.config.get("project", {}).get("data_replicate_shift_windows", 0) > 0
                )
                if df_override is None and has_replicate:
                    df_seed = self._make_replicate_df(df, seed, timestamp_col)
                    splitter_seed = WindowSplitter(self.config["data"])
                    windows_seed = splitter_seed.split(df_seed)
                else:
                    windows_seed = windows
                
                run_results = self._run_single(
                    policy=policy,
                    windows=windows_seed,
                    seed=seed,
                )
                all_results.extend(run_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save to CSV
        output_dir = Path(self.config.get("output", {}).get("results_dir", "results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_filename if output_filename else "experiment_results.csv"
        csv_path = output_dir / fname
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        
        return results_df
    
    def _run_single(
        self,
        policy,
        windows,
        seed: int,
    ) -> List[Dict]:
        """Run a single policy on all windows with a given seed."""
        np.random.seed(seed)
        # NOTE: when using data_replicate_shift in the config, the runner
        # receives different 'windows' objects per seed (shifted offsets),
        # so the model sees genuinely different data replicates.
        
        # Override model seed if applicable
        model_config = copy.deepcopy(self.config["model"])
        if "params" in model_config and isinstance(model_config["params"], dict):
            model_config["params"]["random_state"] = seed
        
        # Initialize fresh components for this run
        trainer = ModelTrainer(model_config)
        
        stat_method = self.config.get("detection", {}).get("statistical_method", "psi")
        if stat_method == "ks":
            detector = KSDriftDetector(
                alpha=self.config.get("detection", {}).get("ks_alpha", 0.05)
            )
        else:
            detector = PSIDriftDetector(
                threshold=self.config.get("detection", {}).get("psi_threshold", 0.2)
            )
        
        shap_computer = ShapComputer(self.config["explainability"], seed=seed)
        mag_tracker = MagnitudeTracker(
            threshold=self.config["explainability"].get(
                "magnitude_threshold", 0.1
            )
        )
        rank_tracker = RankChangeTracker(
            threshold=self.config["explainability"].get(
                "rank_change_threshold", 0.3
            ),
            top_k=self.config["explainability"].get("rank_top_k", 5),
            use_weightedtau=self.config["explainability"].get("use_weightedtau", False),
        )
        dis_computer = DriftImpactScore(
            alpha=self.config["fusion"]["alpha"],
            beta=self.config["fusion"]["beta"],
            gamma=self.config["fusion"]["gamma"],
        )
        use_calibration = self.config.get("fusion", {}).get("use_calibrated_zscore", True)
        signal_normalizer = SignalNormalizer(warmup_windows=2)
        atc = AdaptiveThresholdController(
            lookback=self.config["fusion"].get("adaptive_lookback", 10),
            lambda_val=self.config["fusion"].get("adaptive_lambda", 1.92),
            min_threshold=self.config["fusion"].get("min_threshold", 0.05),
            warmup_threshold=self.config["fusion"].get("warmup_threshold", float("inf")),
            fixed_threshold=self.config["fusion"].get("fixed_threshold", 3.0),
            mode=self.config["fusion"].get("threshold_mode", "fixed"),
        )
        label_delay = self.config.get("project", {}).get("label_delay", 0)
        
        run_results = []
        total_cost = 0.0
        retrain_count = 0
        last_dis_result = {}
        last_atc_result = {}
        run_name = f"{policy.policy_id}_seed{seed}"
        
        def execute_pipeline():
            nonlocal total_cost, retrain_count, last_dis_result, last_atc_result
            
            if mlflow is not None:
                try:
                    mlflow.log_params({
                        "policy": policy.policy_id,
                        "seed": seed,
                        "n_windows": len(windows),
                        "model_type": self.config["model"]["type"],
                    })
                except Exception as e:
                    logger.debug(f"MLflow log_params exception: {e}")
            
            # === WINDOW 0: Initial training (always) ===
            # No accuracy is logged for window 0 — there is no prior model to
            # evaluate before the first training run.  Logging NaN makes it
            # explicit in downstream analysis that window 0 is train-only.
            logger.info(f"Window 0: Initial training only (no prior model — accuracy=NaN)")
            train_metrics = trainer.train(
                windows[0].X, windows[0].y
            )
            total_cost += train_metrics["cost_seconds"]
            retrain_count += 1
            
            # Set drift detection reference
            detector.set_reference(windows[0].X)
            
            # Compute initial SHAP profile and reference feature weights
            shap_result = shap_computer.compute(
                trainer.get_model(), windows[0].X
            )
            noise_floor_w0 = shap_result.get("noise_floor", {})
            mag_tracker.update_and_compare(shap_result["mean_abs_shap"], noise_floor=noise_floor_w0)
            rank_tracker.update_and_compare(shap_result["mean_abs_shap"], noise_floor=noise_floor_w0)
            ref_feature_weights = shap_result["mean_abs_shap"].to_dict()
            
            # Log window 0 — accuracy is NaN (no evaluation possible yet).
            w0_row = {
                "policy": policy.policy_id,
                "policy_name": policy.name,
                "seed": seed,
                "window_id": 0,
                # NaN: honest — no out-of-sample evaluation is possible
                # on the window used for initial training.
                "accuracy": float("nan"),
                "f1": float("nan"),
                "retrained": True,
                "retrain_reason": "Initial training",
                "cost_seconds": train_metrics["cost_seconds"],
                "cumulative_cost": total_cost,
                "cumulative_retrains": retrain_count,
                "retrain_count": retrain_count,
                "stat_drift_score": 0.0,
                "shap_magnitude_score": 0.0,
                "shap_rank_change_score": 0.0,
                "dis": 0.0,
                "threshold": 0.0,
                # Warmup flag: window 0 is always the initial-train window.
                "is_warmup": True,
                # Provenance: is this dataset real or synthetic?
                "data_source": self.config.get("data", {}).get("data_source_type", "synthetic"),
            }
            run_results.append(w0_row)
            
            if mlflow is not None:
                try:
                    mlflow.log_metrics({
                        # Renamed: this is the in-training-window held-out val score,
                        # NOT a prequential evaluation.  Using a distinct key prevents
                        # it being averaged with the out-of-sample w1..N accuracy values.
                        "train_val_accuracy_w0": train_metrics["val_accuracy"],
                        "train_val_f1_w0": train_metrics["val_f1"],
                        "dis_w0": 0.0,
                    }, step=0)
                except Exception as e:
                    logger.debug(f"MLflow log_metrics exception: {e}")
            
            # === WINDOWS 1..N: Evaluate, detect, decide ===
            # Protocol: test-then-train (prequential / interleaved test-then-train).
            # The model is FIRST evaluated on the new window (out-of-sample),
            # THEN optionally retrained on that window.  The logged accuracy is
            # always the PRE-RETRAIN evaluation — the only honest metric.
            for w_idx in range(1, len(windows)):
                window = windows[w_idx]
                logger.info(f"\n--- Window {w_idx} ---")
                
                # 1. EVALUATE current model on this window BEFORE any retrain.
                #    The model has been trained only on windows 0..(w-1), so
                #    this is a genuine out-of-sample test.
                eval_metrics = trainer.evaluate(window.X, window.y)
                logger.info(
                    f"  Pre-retrain eval: acc={eval_metrics['accuracy']:.4f}"
                    f" (out-of-sample, model v{trainer.version})"
                )

                # Delayed metrics for performance trigger if label_delay > 0
                delayed_eval = eval_metrics
                if label_delay > 0:
                    delay_idx = w_idx - label_delay
                    if delay_idx >= 1:
                        delayed_eval = trainer.evaluate(windows[delay_idx].X, windows[delay_idx].y)
                    else:
                        delayed_eval = {"accuracy": float("nan"), "f1_weighted": float("nan")}
                
                # 2. Statistical drift detection
                stat_drift = detector.detect(window.X, feature_weights=ref_feature_weights)
                
                # 3. SHAP computation
                shap_result = shap_computer.compute(
                    trainer.get_model(), window.X
                )
                noise_floor = shap_result.get("noise_floor", {})
                
                # 4. SHAP magnitude tracking (calibrated with noise floor)
                mag_result = mag_tracker.update_and_compare(
                    shap_result["mean_abs_shap"],
                    noise_floor=noise_floor,
                )
                
                # 5. SHAP rank-change tracking (calibrated with noise floor on top features)
                rank_result = rank_tracker.update_and_compare(
                    shap_result["mean_abs_shap"],
                    noise_floor=noise_floor,
                )
                
                # 6. DIS (fused score)
                if use_calibration:
                    dis_result = dis_computer.compute(
                        stat_drift_score=stat_drift.get("z_score", stat_drift["drift_score"]),
                        shap_magnitude_score=mag_result.get("z_score", mag_result["magnitude_score"]),
                        shap_rank_change_score=rank_result.get("z_score", rank_result["rank_change_score"]),
                        window_id=w_idx,
                    )
                else:
                    normalized = signal_normalizer.normalize(
                        stat=stat_drift["drift_score"],
                        mag=mag_result["magnitude_score"],
                        rank=rank_result["rank_change_score"],
                    )
                    dis_result = dis_computer.compute(
                        stat_drift_score=normalized["stat"],
                        shap_magnitude_score=normalized["magnitude"],
                        shap_rank_change_score=normalized["rank_change"],
                        window_id=w_idx,
                    )
                
                # 7. Adaptive threshold decision
                atc_result = atc.update_and_decide(
                    dis_value=dis_result["dis"],
                    window_id=w_idx,
                )
                last_dis_result = dis_result
                last_atc_result = atc_result
                
                # 8. Policy decision
                decision = policy.decide(
                    window_id=w_idx,
                    stat_drift=stat_drift,
                    shap_magnitude=mag_result,
                    shap_rank_change=rank_result,
                    dis_result=dis_result,
                    atc_result=atc_result,
                    eval_metrics=delayed_eval,
                    trainer=trainer,
                    window=window,
                    shap_computer=shap_computer,
                )
                
                # 9. If the policy triggers a retrain, do it AFTER logging
                #    eval_metrics (which are already out-of-sample and honest).
                #    We do NOT overwrite eval_metrics after retraining — doing
                #    so would produce in-sample accuracy and invalidate all
                #    cross-policy comparisons.
                retrain_cost = 0.0
                if decision.should_retrain:
                    train_metrics = trainer.train(
                        window.X, window.y
                    )
                    retrain_cost = train_metrics["cost_seconds"]
                    total_cost += retrain_cost
                    retrain_count += 1
                    
                    # Update drift reference to the newly trained window
                    detector.set_reference(window.X)
                    
                    # Reset SHAP trackers so next window is compared to
                    # the new model's SHAP profile, not the old one.
                    mag_tracker.reset()
                    rank_tracker.reset()

                    # Neither the normalizer nor the ATC is reset on retrain.
                    # - Normalizer: signal scales (KS stat, SHAP L1/2, Spearman rho)
                    #   are properties of the data/feature set, not the model.
                    # - ATC: must accumulate DIS history across epochs to exit warmup.
                    
                    # Re-compute SHAP for new model baseline
                    shap_result = shap_computer.compute(
                        trainer.get_model(), window.X
                    )
                    noise_floor_post = shap_result.get("noise_floor", {})
                    mag_tracker.update_and_compare(
                        shap_result["mean_abs_shap"],
                        noise_floor=noise_floor_post,
                    )
                    rank_tracker.update_and_compare(
                        shap_result["mean_abs_shap"],
                        noise_floor=noise_floor_post,
                    )
                    ref_feature_weights = shap_result["mean_abs_shap"].to_dict()
                    
                    logger.info(
                        f"  RETRAINED: cost={retrain_cost:.2f}s "
                        f"(honest pre-retrain acc already logged: "
                        f"{eval_metrics['accuracy']:.4f})"
                    )
                
                # 10. Log everything.
                #     eval_metrics here is ALWAYS the pre-retrain, out-of-sample
                #     evaluation.  It was computed BEFORE any retrain on this window.
                row = {
                    "policy": policy.policy_id,
                    "policy_name": policy.name,
                    "seed": seed,
                    "window_id": w_idx,
                    # Honest out-of-sample accuracy: evaluated BEFORE any retrain
                    "accuracy": eval_metrics["accuracy"],
                    "f1": eval_metrics["f1_weighted"],
                    "retrained": decision.should_retrain,
                    "retrain_reason": decision.reason,
                    "cost_seconds": retrain_cost,
                    "cumulative_cost": total_cost,
                    "cumulative_retrains": retrain_count,
                    # Primary cost metric: count of retrains (timing-noise-free)
                    "retrain_count": retrain_count,
                    "stat_drift_score": stat_drift["drift_score"],
                    "shap_magnitude_score": mag_result["magnitude_score"],
                    "shap_rank_change_score": rank_result["rank_change_score"],
                    "dis": dis_result["dis"],
                    "threshold": atc_result["threshold"],
                    # Whether the ATC was in warmup mode for this window.
                    # True means the trigger came from the fixed warmup_threshold
                    # (0.1), NOT from the adaptive μ+λσ formula.  Used downstream
                    # to count warmup-driven vs truly adaptive retrains.
                    "is_warmup": atc_result.get("is_warmup", False),
                    # Fusion hyperparameters — logged for full reproducibility
                    "alpha": self.config["fusion"]["alpha"],
                    "beta": self.config["fusion"]["beta"],
                    "gamma": self.config["fusion"]["gamma"],
                    "adaptive_lambda": self.config["fusion"].get("adaptive_lambda", 1.5),
                    "lookback_k": self.config["fusion"].get("adaptive_lookback", 3),
                    # Provenance: is this dataset real or synthetic?
                    "data_source": self.config.get("data", {}).get("data_source_type", "synthetic"),
                }
                run_results.append(row)
                
                if mlflow is not None:
                    try:
                        mlflow.log_metrics({
                            f"accuracy_w{w_idx}": eval_metrics["accuracy"],
                            f"f1_w{w_idx}": eval_metrics["f1_weighted"],
                            f"dis_w{w_idx}": dis_result["dis"],
                        }, step=w_idx)
                    except Exception as e:
                        logger.debug(f"MLflow log_metrics step exception: {e}")
            
            # Log final summary to MLflow
            if mlflow is not None:
                try:
                    mlflow.log_metrics({
                        "total_cost": total_cost,
                        "total_retrains": retrain_count,
                        "final_accuracy": run_results[-1]["accuracy"],
                        # Use nanmean: window 0 has accuracy=NaN (no prior model),
                        # so plain np.mean would propagate NaN to the summary metric.
                        "mean_accuracy": float(np.nanmean([r["accuracy"] for r in run_results])),
                    })
                except Exception as e:
                    logger.debug(f"MLflow summary log exception: {e}")

        if mlflow is not None:
            try:
                with mlflow.start_run(run_name=run_name, nested=True):
                    execute_pipeline()
            except Exception:
                execute_pipeline()
        else:
            execute_pipeline()

        # Persist latest model and DIS status to results/ using ModelRegistry
        try:
            from driftx.training.registry import ModelRegistry
            output_dir = self.config.get("output", {}).get("results_dir", "results")
            registry = ModelRegistry(output_dir)
            registry.export_latest_artifacts(
                trainer=trainer,
                dis_result=last_dis_result,
                atc_result=last_atc_result,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.debug(f"ModelRegistry export notice: {e}")
        
        return run_results
    
    def _get_seeds(self) -> List[int]:
        """Generate seed list from config."""
        base_seed = self.config.get("project", {}).get("seed", 42)
        n_seeds = self.config.get("project", {}).get("n_seeds", 1)
        return [base_seed + i for i in range(n_seeds)]

    def _make_replicate_df(
        self,
        df: pd.DataFrame,
        seed: int,
        timestamp_col: str,
    ) -> pd.DataFrame:
        """Offset the window grid by dropping leading rows per seed.

        This creates genuinely different data replicates per seed by sliding the
        window boundaries across the observation sequence, so each seed trains
        and evaluates on a different temporal slice rather than just varying
        the model seed.

        Args:
            df: Full dataset, already sorted chronologically.
            seed: Current seed value.
            timestamp_col: Timestamp column name.

        Returns:
            Shifted DataFrame with leading rows dropped.
        """
        base = self.config["project"].get("seed", 42)
        step = self.config["project"].get("data_replicate_offset_rows", 0)
        if step == 0 and self.config["project"].get("data_replicate_shift_windows", 0) > 0:
            samples_per_window = self.config.get("data", {}).get("min_window_samples", 1000)
            step = int(self.config["project"].get("data_replicate_shift_windows", 0) * samples_per_window)
        if step == 0 or seed == base:
            return df

        offset = (seed - base) * step
        if offset < len(df):
            logger.info(
                f"Data replicate offset applied for seed {seed}: "
                f"dropped {offset} leading rows (step={step})"
            )
            return df.iloc[offset:].reset_index(drop=True)
        return df
