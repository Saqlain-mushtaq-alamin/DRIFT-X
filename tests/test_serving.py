"""Unit tests for DRIFT-X Model Registry and FastAPI Serving Layer."""
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from starlette.testclient import TestClient

from driftx.training.registry import ModelRegistry
from driftx.serving.api import app, set_artifact_paths, reload_model_artifact, ModelServer


@pytest.fixture
def tmp_artifacts(tmp_path):
    """Fixture providing temporary model and DIS artifact paths."""
    model_path = tmp_path / "latest_model.joblib"
    dis_path = tmp_path / "latest_dis.json"
    return tmp_path, model_path, dis_path


def test_registry_model_save_load(tmp_artifacts):
    tmp_path, model_path, _ = tmp_artifacts
    registry = ModelRegistry(base_dir=tmp_path)

    # Train dummy model
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
    y = pd.Series([0, 1, 0])
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit(X, y)

    saved_path = registry.save_model(model, path=model_path, metadata={"version": 3})
    assert saved_path.exists()
    assert (tmp_path / "latest_model.json").exists()

    loaded_model = registry.load_model(path=model_path)
    preds = loaded_model.predict(X)
    assert len(preds) == 3
    assert (preds == 1).all()


def test_registry_drift_status_save_load(tmp_artifacts):
    tmp_path, _, dis_path = tmp_artifacts
    registry = ModelRegistry(base_dir=tmp_path)

    sample_status = {
        "dis": 0.45,
        "stat_drift": 0.35,
        "threshold": 0.40,
        "is_drifting": True,
        "np_scalar": np.float64(0.85),
    }
    saved_path = registry.save_drift_status(sample_status, path=dis_path)
    assert saved_path.exists()

    loaded = registry.load_drift_status(path=dis_path)
    assert loaded["dis"] == 0.45
    assert loaded["is_drifting"] is True
    assert loaded["np_scalar"] == 0.85


def test_api_health_endpoint_no_model(tmp_artifacts):
    _, model_path, dis_path = tmp_artifacts
    set_artifact_paths(str(model_path), str(dis_path))
    reload_model_artifact()

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False
        assert data["model_version"] == 0


def test_api_predict_without_model(tmp_artifacts):
    _, model_path, dis_path = tmp_artifacts
    set_artifact_paths(str(model_path), str(dis_path))
    reload_model_artifact()

    with TestClient(app) as client:
        resp = client.post("/predict", json={"features": {"f1": 1.0, "f2": 2.0}})
        assert resp.status_code == 503
        assert "Model not loaded" in resp.json()["detail"]


def test_api_predict_with_model(tmp_artifacts):
    tmp_path, model_path, dis_path = tmp_artifacts
    registry = ModelRegistry(base_dir=tmp_path)

    # Train dummy model with predict_proba
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    y = pd.Series([0, 1])
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)
    registry.save_model(model, path=model_path, metadata={"version": 2})

    set_artifact_paths(str(model_path), str(dis_path))
    reload_model_artifact()

    with TestClient(app) as client:
        # Check health with loaded model
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["model_loaded"] is True
        assert health_resp.json()["model_version"] == 2

        # Make prediction
        pred_resp = client.post("/predict", json={"features": {"f1": 1.5, "f2": 3.5}})
        assert pred_resp.status_code == 200
        data = pred_resp.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["model_version"] == 2
        assert len(data["probability"]) == 2


def test_api_drift_status_endpoint(tmp_artifacts):
    tmp_path, model_path, dis_path = tmp_artifacts
    registry = ModelRegistry(base_dir=tmp_path)
    set_artifact_paths(str(model_path), str(dis_path))

    with TestClient(app) as client:
        # Status before file exists
        resp_empty = client.get("/drift-status")
        assert resp_empty.status_code == 200
        assert resp_empty.json() == {"status": "no drift data available"}

        # Write status file
        registry.save_drift_status({"status": "active", "dis": 0.32, "window_id": 4}, path=dis_path)
        resp_data = client.get("/drift-status")
        assert resp_data.status_code == 200
        assert resp_data.json()["dis"] == 0.32
        assert resp_data.json()["window_id"] == 4


def test_model_server_wrapper(tmp_artifacts):
    _, model_path, dis_path = tmp_artifacts
    server = ModelServer(host="127.0.0.1", port=8000, model_path=str(model_path), dis_path=str(dis_path))
    assert server.app is not None
    assert server.load() is False
