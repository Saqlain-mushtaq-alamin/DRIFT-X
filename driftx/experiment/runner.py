"""
Main experiment orchestrator — runs the full DRIFT-X pipeline.

For each policy × each seed:
  1. Load data and create windows
  2. Train initial model on window 0
  3. For each subsequent window:
     a. Evaluate current model (accuracy, F1)
     b. Compute statistical drift
     c. Compute SHAP values → magnitude + rank-change
     d. Compute DIS (fused score)
     e. Ask the active policy: retrain?
     f. If yes: retrain, log cost
     g. Log everything to MLflow + CSV
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
        
        for policy_id, policy in policies.items():
            for seed in seeds:
                logger.info(f"\n{'='*60}")
                logger.info(f"Policy: {policy.name} | Seed: {seed}")
                logger.info(f"{'='*60}")
                
                run_results = self._run_single(
                    policy=policy,
                    windows=windows,
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
            )
        )
        dis_computer = DriftImpactScore(
            alpha=self.config["fusion"]["alpha"],
            beta=self.config["fusion"]["beta"],
            gamma=self.config["fusion"]["gamma"],
        )
        # Per-signal normalizer: scales each DIS input to [0, 1] so the
        # fused score occupies the full range and can exceed the warmup threshold.
        signal_normalizer = SignalNormalizer(warmup_windows=2)
        atc = AdaptiveThresholdController(
            lookback=self.config["fusion"].get("adaptive_lookback", 3),
            lambda_val=self.config["fusion"].get("adaptive_lambda", 1.5),
        )
        
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
            logger.info(f"Window 0: Initial training")
            train_metrics = trainer.train(
                windows[0].X, windows[0].y
            )
            total_cost += train_metrics["cost_seconds"]
            retrain_count += 1
            
            # Set drift detection reference
            detector.set_reference(windows[0].X)
            
            # Compute initial SHAP profile
            shap_result = shap_computer.compute(
                trainer.get_model(), windows[0].X
            )
            mag_tracker.update_and_compare(shap_result["mean_abs_shap"])
            rank_tracker.update_and_compare(shap_result["mean_abs_shap"])
            
            # Log window 0
            w0_row = {
                "policy": policy.policy_id,
                "policy_name": policy.name,
                "seed": seed,
                "window_id": 0,
                "accuracy": train_metrics["val_accuracy"],
                "f1": train_metrics["val_f1"],
                "retrained": True,
                "retrain_reason": "Initial training",
                "cost_seconds": train_metrics["cost_seconds"],
                "cumulative_cost": total_cost,
                "cumulative_retrains": retrain_count,
                "stat_drift_score": 0.0,
                "shap_magnitude_score": 0.0,
                "shap_rank_change_score": 0.0,
                "dis": 0.0,
                "threshold": 0.0,
            }
            run_results.append(w0_row)
            
            if mlflow is not None:
                try:
                    mlflow.log_metrics({
                        "accuracy_w0": train_metrics["val_accuracy"],
                        "f1_w0": train_metrics["val_f1"],
                        "dis_w0": 0.0,
                    }, step=0)
                except Exception as e:
                    logger.debug(f"MLflow log_metrics exception: {e}")
            
            # === WINDOWS 1..N: Evaluate, detect, decide ===
            for w_idx in range(1, len(windows)):
                window = windows[w_idx]
                logger.info(f"\n--- Window {w_idx} ---")
                
                # 1. Evaluate current model on this window
                eval_metrics = trainer.evaluate(window.X, window.y)
                
                # 2. Statistical drift detection
                stat_drift = detector.detect(window.X)
                
                # 3. SHAP computation
                shap_result = shap_computer.compute(
                    trainer.get_model(), window.X
                )
                
                # 4. SHAP magnitude tracking
                mag_result = mag_tracker.update_and_compare(
                    shap_result["mean_abs_shap"]
                )
                
                # 5. SHAP rank-change tracking
                rank_result = rank_tracker.update_and_compare(
                    shap_result["mean_abs_shap"]
                )
                
                # 6. Normalise signals to [0,1] then compute DIS (fused score)
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
                )
                
                # 9. If retrain, do it
                retrain_cost = 0.0
                if decision.should_retrain:
                    train_metrics = trainer.train(
                        window.X, window.y
                    )
                    retrain_cost = train_metrics["cost_seconds"]
                    total_cost += retrain_cost
                    retrain_count += 1
                    
                    # Re-evaluate after retraining
                    eval_metrics = trainer.evaluate(window.X, window.y)
                    
                    # Update drift reference to new model dataset
                    detector.set_reference(window.X)
                    
                    # Reset SHAP trackers
                    mag_tracker.reset()
                    rank_tracker.reset()

                    # Neither the normalizer nor the ATC is reset on retrain.
                    # - Normalizer: signal scales (KS stat, SHAP L1/2, Spearman rho)
                    #   are properties of the data/feature set, not the model.
                    #   Resetting on retrain introduced two-window warmup gaps where
                    #   DIS=raw-values (tiny), collapsing the ATC history to zeros
                    #   and driving theta to min_threshold (false Pareto collapse).
                    # - ATC: must accumulate DIS history across epochs to exit warmup.
                    #   Post-retrain DIS values (low drift, just trained) are valid
                    #   signal that rightfully pull the adaptive threshold downward.
                    
                    # Re-compute SHAP for new model
                    shap_result = shap_computer.compute(
                        trainer.get_model(), window.X
                    )
                    mag_tracker.update_and_compare(
                        shap_result["mean_abs_shap"]
                    )
                    rank_tracker.update_and_compare(
                        shap_result["mean_abs_shap"]
                    )
                    
                    logger.info(
                        f"  RETRAINED: cost={retrain_cost:.2f}s, "
                        f"new_acc={eval_metrics['accuracy']:.4f}"
                    )
                
                # 10. Log everything
                row = {
                    "policy": policy.policy_id,
                    "policy_name": policy.name,
                    "seed": seed,
                    "window_id": w_idx,
                    "accuracy": eval_metrics["accuracy"],
                    "f1": eval_metrics["f1_weighted"],
                    "retrained": decision.should_retrain,
                    "retrain_reason": decision.reason,
                    "cost_seconds": retrain_cost,
                    "cumulative_cost": total_cost,
                    "cumulative_retrains": retrain_count,
                    "stat_drift_score": stat_drift["drift_score"],
                    "shap_magnitude_score": mag_result["magnitude_score"],
                    "shap_rank_change_score": rank_result["rank_change_score"],
                    "dis": dis_result["dis"],
                    "threshold": atc_result["threshold"],
                    # Fusion hyperparameters — logged for full reproducibility (Bug 6 fix)
                    "alpha": self.config["fusion"]["alpha"],
                    "beta": self.config["fusion"]["beta"],
                    "gamma": self.config["fusion"]["gamma"],
                    "adaptive_lambda": self.config["fusion"].get("adaptive_lambda", 1.5),
                    "lookback_k": self.config["fusion"].get("adaptive_lookback", 3),
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
                        "mean_accuracy": float(np.mean([r["accuracy"] for r in run_results])),
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
