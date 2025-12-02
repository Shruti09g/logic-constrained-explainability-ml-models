# src/utils.py
import json
import os
from typing import Any, Dict

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    if path is None or path == "":
        return
    os.makedirs(path, exist_ok=True)

def save_json(obj: Any, path: str):
    """Save an object as JSON to path (creates parent dirs)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    """Load JSON from path."""
    with open(path, "r") as f:
        return json.load(f)

def series_to_jsonlike(series):
    """Convert pandas Series to JSON-serializable dict (cast numpy types)."""
    result = {}
    for k, v in series.items():
        try:
            import numpy as _np
            if isinstance(v, (_np.generic, )):
                v = v.item()
        except Exception:
            pass
        result[k] = v
    return result
