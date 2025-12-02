# src/xreason_interface.py
import subprocess
import json
import os
from utils import save_json, load_json, ensure_dir

# path to the minimal xreason runner (interval-based)
XREASON_CMD = "/home/vivresavie/xreason/src/xreason.py"

def call_xreason_get_formal_region(model_path: str, instance_json_path: str, out_json_path: str, timeout: int = 60):
    ensure_dir(os.path.dirname(out_json_path) or ".")
    if not os.path.exists(XREASON_CMD):
        raise FileNotFoundError(f"XReason runner not found at {XREASON_CMD}")
    cmd = ["python3", XREASON_CMD, "--model", model_path, "--instance", instance_json_path, "--out", out_json_path]
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"XReason failed: {e}")
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"XReason timed out: {e}")

    if not os.path.exists(out_json_path):
        raise RuntimeError("XReason did not produce output file")
    return load_json(out_json_path)

def parse_xreason_region(xreason_out: dict):
    """
    Parse the interval-format region emitted by xreason.py.
    Returns canonical dict: raw_feature -> ["eq", value] OR [low, high]
    """
    raw = xreason_out.get("region", {}) or {}
    parsed = {}
    for k, v in raw.items():
        if isinstance(v, list) and len(v) == 2:
            left, right = v[0], v[1]
            # Try numeric interval
            try:
                lo = float(v[0])
                hi = float(v[1])
                parsed[k] = [lo, hi]
            except Exception:
                # treat as equality if not numeric
                if left == "eq":
                    parsed[k] = ["eq", right]
                else:
                    parsed[k] = ["eq", right]
        else:
            parsed[k] = ["eq", v]
    return parsed
