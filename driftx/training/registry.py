"""Model registry and artifact persistence module for DRIFT-X."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import joblib

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages serialization, versioning, and loading of trained models and DIS status."""

    def __init__(self, base_dir: Union[str, Path] = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        path: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Serialize model artifact to disk using joblib.

        Args:
            model: Trained classifier model instance.
            path: Target file path. Defaults to '<base_dir>/latest_model.joblib'.
            metadata: Optional dictionary with model hyperparameters, version, metrics.

        Returns:
            Path: Resolved Path to saved model file.
        """
        target_path = Path(path) if path else self.base_dir / "latest_model.joblib"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, target_path)
        logger.info(f"Saved model artifact to {target_path}")

        if metadata:
            meta_path = target_path.with_suffix(".json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved model metadata to {meta_path}")

        return target_path

    def load_model(self, path: Optional[Union[str, Path]] = None) -> Any:
        """
        Load model artifact from disk.

        Args:
            path: Path to model file. Defaults to '<base_dir>/latest_model.joblib'.

        Returns:
            Deserialized model instance.
        """
        target_path = Path(path) if path else self.base_dir / "latest_model.joblib"
        if not target_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {target_path}")
        model = joblib.load(target_path)
        logger.info(f"Loaded model artifact from {target_path}")
        return model

    def save_drift_status(
        self,
        dis_data: Dict[str, Any],
        path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Persist latest Drift Impact Score (DIS) and threshold monitoring state.

        Args:
            dis_data: Dictionary of DIS scores, individual signals, and threshold status.
            path: Target JSON file path. Defaults to '<base_dir>/latest_dis.json'.

        Returns:
            Path: Resolved Path to saved JSON file.
        """
        target_path = Path(path) if path else self.base_dir / "latest_dis.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure all values are JSON serializable
        clean_data = {}
        for k, v in dis_data.items():
            if hasattr(v, "item"):  # numpy scalars
                clean_data[k] = v.item()
            elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                clean_data[k] = v
            else:
                clean_data[k] = str(v)

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2)
        logger.info(f"Saved drift status to {target_path}")
        return target_path

    def load_drift_status(self, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Read latest DIS drift status.

        Args:
            path: Path to status JSON file. Defaults to '<base_dir>/latest_dis.json'.

        Returns:
            Dict containing monitoring status or fallback message.
        """
        target_path = Path(path) if path else self.base_dir / "latest_dis.json"
        if not target_path.exists():
            return {"status": "no drift data available"}
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def export_latest_artifacts(
        self,
        trainer: Any,
        dis_result: Optional[Dict[str, Any]] = None,
        atc_result: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Convenience method to export both latest_model.joblib and latest_dis.json."""
        out_path = Path(output_dir) if output_dir else self.base_dir
        out_path.mkdir(parents=True, exist_ok=True)

        if hasattr(trainer, "get_model"):
            model = trainer.get_model()
            if model is not None:
                self.save_model(
                    model=model,
                    path=out_path / "latest_model.joblib",
                    metadata={"version": getattr(trainer, "version", 1)},
                )

        status_payload: Dict[str, Any] = {
            "status": "active",
        }
        if dis_result:
            status_payload.update({
                "dis": dis_result.get("dis", 0.0),
                "stat_drift": dis_result.get("stat_drift", 0.0),
                "shap_magnitude": dis_result.get("shap_magnitude", 0.0),
                "shap_rank_change": dis_result.get("shap_rank_change", 0.0),
                "window_id": dis_result.get("window_id", 0),
            })
        if atc_result:
            status_payload.update({
                "threshold": atc_result.get("threshold", 0.0),
                "is_drifting": atc_result.get("exceeds_threshold", False),
                "in_warmup": atc_result.get("in_warmup", False),
            })

        self.save_drift_status(status_payload, path=out_path / "latest_dis.json")


# Global default registry instance
default_registry = ModelRegistry()


def save_model(model: Any, path: Optional[Union[str, Path]] = None, metadata: Optional[Dict[str, Any]] = None) -> Path:
    return default_registry.save_model(model, path, metadata)


def load_model(path: Optional[Union[str, Path]] = None) -> Any:
    return default_registry.load_model(path)


def save_drift_status(dis_data: Dict[str, Any], path: Optional[Union[str, Path]] = None) -> Path:
    return default_registry.save_drift_status(dis_data, path)


def load_drift_status(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    return default_registry.load_drift_status(path)
