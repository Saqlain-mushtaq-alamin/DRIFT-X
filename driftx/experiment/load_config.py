"""Unified configuration loader for DRIFT-X.

Ensures all scripts load from configs/frozen.yaml with standardized overrides.
"""
import copy
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict b into dict a."""
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = _merge(a.get(k, {}), v)
        else:
            a[k] = v
    return a


def load_frozen(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load base configuration from configs/frozen.yaml (falls back to default.yaml)
    and apply deep overrides.
    """
    cfg_path = ROOT / "configs" / "frozen.yaml"
    if not cfg_path.exists():
        cfg_path = ROOT / "configs" / "default.yaml"
    
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    return _merge(copy.deepcopy(cfg), overrides or {})
