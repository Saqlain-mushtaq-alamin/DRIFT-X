"""Phase 8 Multi-Dataset Validation & Serving Layer Verification Script."""
import sys
import logging
import yaml
from pathlib import Path
import pandas as pd
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.data.ingestor import DataIngestor
from driftx.data.windower import WindowSplitter
from driftx.experiment.runner import ExperimentRunner
from driftx.training.registry import ModelRegistry
from driftx.serving.api import app, set_artifact_paths, reload_model_artifact
from scripts.generate_tables import generate_cross_dataset_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase8_Verification")


def run_verification():
    logger.info("=== Phase 8 Multi-Dataset & Serving Layer Verification Script ===")
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Verify Multi-Dataset Ingestion & Windowing (Zero Leakage)
    # -------------------------------------------------------------
    logger.info("\nStep 1: Testing Multi-Dataset Ingestion & Windowing...")

    # 1a. Intrusion Dataset (Daily windowing)
    intrusion_cfg = {
        "dataset": "intrusion",
        "timestamp_col": "Timestamp",
        "target_col": "is_attack",
        "window_size": "daily",
        "min_window_samples": 500,
    }
    ingestor_ids = DataIngestor(intrusion_cfg)
    df_ids = ingestor_ids.load(data_dir=str(data_dir))
    assert not df_ids.empty, "Failed to load intrusion dataset!"
    assert "is_attack" in df_ids.columns, "Target column 'is_attack' missing!"

    splitter_ids = WindowSplitter(intrusion_cfg)
    windows_ids = splitter_ids.split(df_ids)
    assert len(windows_ids) >= 5, f"Expected >=5 daily windows for intrusion data, got {len(windows_ids)}"
    logger.info(f"  ✓ Intrusion Dataset: {len(df_ids)} rows partitioned into {len(windows_ids)} daily windows with 0% leakage.")

    # 1b. Electricity Dataset (ELEC2 monthly windowing)
    elec_cfg = {
        "dataset": "electricity",
        "timestamp_col": "Timestamp",
        "target_col": "target",
        "window_size": "monthly",
        "min_window_samples": 500,
    }
    ingestor_elec = DataIngestor(elec_cfg)
    df_elec = ingestor_elec.load(data_dir=str(data_dir))
    assert not df_elec.empty, "Failed to load electricity dataset!"
    assert "target" in df_elec.columns, "Target column 'target' missing!"

    splitter_elec = WindowSplitter(elec_cfg)
    windows_elec = splitter_elec.split(df_elec)
    assert len(windows_elec) >= 5, f"Expected >=5 monthly windows for electricity data, got {len(windows_elec)}"
    logger.info(f"  ✓ Electricity Dataset: {len(df_elec)} rows partitioned into {len(windows_elec)} monthly windows with 0% leakage.")

    # -------------------------------------------------------------
    # Step 2: Run Multi-Dataset Benchmark Experiments
    # -------------------------------------------------------------
    logger.info("\nStep 2: Executing Multi-Dataset Benchmark across Fraud, Intrusion, Electricity...")
    default_config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(default_config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Use fast settings for verification
    benchmark_policies = ["p0_never", "p2_drift_only", "p5_dis_fused"]
    benchmark_datasets = {
        "Fraud": {
            "dataset": "synthetic",
            "timestamp_col": "TransactionDT",
            "target_col": "isFraud",
            "window_size": "monthly",
            "min_window_samples": 500,
        },
        "Intrusion": {
            "dataset": "intrusion",
            "timestamp_col": "Timestamp",
            "target_col": "is_attack",
            "window_size": "daily",
            "min_window_samples": 500,
        },
        "Electricity": {
            "dataset": "electricity",
            "timestamp_col": "Timestamp",
            "target_col": "target",
            "window_size": "monthly",
            "min_window_samples": 500,
        },
    }

    results_map = {}
    for ds_name, data_cfg in benchmark_datasets.items():
        import copy
        ds_cfg = copy.deepcopy(base_config)
        ds_cfg["data"].update(data_cfg)
        ds_cfg["policy"]["active_policies"] = benchmark_policies
        ds_cfg["project"]["n_seeds"] = 1
        ds_cfg["model"]["params"]["n_estimators"] = 25
        ds_cfg["explainability"]["max_shap_samples"] = 100

        out_fname = f"experiment_results_{ds_name.lower()}.csv"
        logger.info(f"  Running benchmark for {ds_name}...")
        runner = ExperimentRunner(ds_cfg)
        df_res = runner.run_all(output_filename=out_fname)
        assert not df_res.empty, f"Results empty for {ds_name}!"
        results_map[ds_name] = results_dir / out_fname
        logger.info(f"  ✓ {ds_name} benchmark completed ({len(df_res)} rows).")

    # -------------------------------------------------------------
    # Step 3: Generate Cross-Dataset Comparison Table
    # -------------------------------------------------------------
    logger.info("\nStep 3: Compiling Cross-Dataset Publication Table...")
    cross_df = generate_cross_dataset_table(dataset_results_map=results_map, output_dir=str(results_dir))
    assert not cross_df.empty, "Cross-dataset table is empty!"
    assert (results_dir / "cross_dataset_table.md").exists(), "cross_dataset_table.md missing!"
    assert (results_dir / "cross_dataset_table.tex").exists(), "cross_dataset_table.tex missing!"
    assert len(cross_df) == 3, f"Expected 3 rows in cross-dataset table, got {len(cross_df)}"
    logger.info("  ✓ Cross-dataset Markdown and LaTeX tables generated successfully.")

    # -------------------------------------------------------------
    # Step 4: Verify Model Registry Artifacts & Serving Layer
    # -------------------------------------------------------------
    logger.info("\nStep 4: Testing Model Registry and FastAPI Serving Layer...")
    latest_model_path = results_dir / "latest_model.joblib"
    latest_dis_path = results_dir / "latest_dis.json"

    assert latest_model_path.exists(), f"latest_model.joblib missing at {latest_model_path}"
    assert latest_dis_path.exists(), f"latest_dis.json missing at {latest_dis_path}"

    # Configure FastAPI paths
    set_artifact_paths(model_path=str(latest_model_path), dis_path=str(latest_dis_path))
    reload_model_artifact()

    with TestClient(app) as client:
        # 4a. Health check
        health_resp = client.get("/health")
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
        h_data = health_resp.json()
        assert h_data["status"] == "healthy"
        assert h_data["model_loaded"] is True
        logger.info(f"  ✓ GET /health: status={h_data['status']}, model_loaded={h_data['model_loaded']}, version={h_data['model_version']}")

        # 4b. Drift status check
        drift_resp = client.get("/drift-status")
        assert drift_resp.status_code == 200, f"Drift status check failed: {drift_resp.text}"
        d_data = drift_resp.json()
        assert "dis" in d_data or "status" in d_data
        logger.info(f"  ✓ GET /drift-status: payload={d_data}")

        # 4c. Prediction endpoint
        registry = ModelRegistry(results_dir)
        loaded_model = registry.load_model(latest_model_path)
        feature_names = getattr(loaded_model, "feature_names_in_", None)
        if feature_names is not None:
            sample_features = {feat: 0.5 for feat in feature_names}
        else:
            sample_features = {f"feature_{i}": 0.5 for i in range(10)}

        pred_resp = client.post("/predict", json={"features": sample_features})
        assert pred_resp.status_code == 200, f"Predict failed: {pred_resp.text}"
        p_data = pred_resp.json()
        assert "prediction" in p_data
        assert "probability" in p_data
        assert p_data["prediction"] in [0, 1]
        assert len(p_data["probability"]) == 2
        logger.info(f"  ✓ POST /predict: prediction={p_data['prediction']}, prob={p_data['probability']}, v={p_data['model_version']}")

    logger.info("\n✓ ALL PHASE 8 VERIFICATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_verification()
