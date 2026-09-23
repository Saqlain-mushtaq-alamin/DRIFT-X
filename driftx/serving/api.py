"""FastAPI serving layer and ModelServer for the DRIFT-X framework."""
import os
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Global model state
_model: Optional[Any] = None
_model_version: int = 0
_model_path: Path = Path("results/latest_model.joblib")
_dis_path: Path = Path("results/latest_dis.json")


def set_artifact_paths(model_path: Optional[str] = None, dis_path: Optional[str] = None) -> None:
    """Override default artifact paths for testing or production deployment."""
    global _model_path, _dis_path
    if model_path:
        _model_path = Path(model_path)
    if dis_path:
        _dis_path = Path(dis_path)


def reload_model_artifact() -> bool:
    """Attempt to load the model from disk into global memory."""
    global _model, _model_version, _model_path
    path = _model_path
    if path.exists():
        try:
            _model = joblib.load(path)
            meta_path = path.with_suffix(".json")
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        _model_version = int(meta.get("version", 1))
                except Exception:
                    _model_version = 1
            else:
                _model_version = 1
            logger.info(f"Loaded model v{_model_version} from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            _model = None
            _model_version = 0
            return False
    else:
        logger.warning(f"Model file not found at {path}")
        _model = None
        _model_version = 0
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load model on startup."""
    reload_model_artifact()
    yield


app = FastAPI(
    title="DRIFT-X Model Server",
    description="Serves the latest drift-adaptive classification model with live drift monitoring.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """Input features dictionary for classification."""
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    """Model prediction output with class, probabilities, and model version."""
    prediction: int
    probability: List[float]
    model_version: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: int


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint confirming service status and model loading."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        model_version=_model_version,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate inference prediction from input feature payload."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure trained model is saved in results/latest_model.joblib or call /model/reload."
        )

    if not request.features:
        raise HTTPException(status_code=400, detail="Feature payload cannot be empty.")

    try:
        features_df = pd.DataFrame([request.features])
        prediction_val = int(_model.predict(features_df)[0])

        if hasattr(_model, "predict_proba"):
            probs = _model.predict_proba(features_df)[0].tolist()
        else:
            probs = [float(1 - prediction_val), float(prediction_val)]

        return PredictionResponse(
            prediction=prediction_val,
            probability=probs,
            model_version=_model_version,
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/drift-status")
async def drift_status():
    """Return the latest Drift Impact Score (DIS) computation and monitoring log."""
    if _dis_path.exists():
        try:
            with open(_dis_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"status": f"error reading drift log: {e}"}
    return {"status": "no drift data available"}


@app.post("/model/reload")
async def reload_model():
    """Manually trigger reloading the latest model artifact from disk."""
    success = reload_model_artifact()
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Could not load model from {_model_path}"
        )
    return {
        "status": "reloaded",
        "model_loaded": True,
        "model_version": _model_version,
    }


class ModelServer:
    """Programmatic wrapper around the FastAPI application for testing or embedded execution."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000, model_path: Optional[str] = None, dis_path: Optional[str] = None):
        self.host = host
        self.port = port
        self.app = app
        if model_path or dis_path:
            set_artifact_paths(model_path=model_path, dis_path=dis_path)

    def load(self) -> bool:
        """Synchronously load model artifacts into memory."""
        return reload_model_artifact()

    def run(self) -> None:
        """Run the server using uvicorn."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
