"""Config hashing and provenance utilities for reproducible DRIFT-X experiments."""
import json
import hashlib
from typing import Dict, Any
from datetime import datetime, timezone


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute a deterministic 12-char SHA-256 hash of experiment configuration.
    
    Excludes non-reproducibility metadata such as output paths and timestamps.
    """
    filtered = {}
    for k, v in config.items():
        if k in ("output", "run_timestamp", "mlflow"):
            continue
        filtered[k] = v
    canonical = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def get_run_timestamp() -> str:
    """Return ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()
