# src/run_refinement.py
"""
Command-line wrapper to run region-constrained explainer refinement for one instance.

Usage:
  python3 src/run_refinement.py --model models/rf_adult.joblib --index 0 --kind lime --samples 500 --r2 0.9
"""

import argparse
import joblib
import os
import pandas as pd
from utils import ensure_dir, save_json, series_to_jsonlike
from xreason_interface import call_xreason_get_formal_region, parse_xreason_region
from refined_explainer import refine_explanation_loop

TMP_DIR = "/home/vivresavie/explainers-project/temp"
ensure_dir(TMP_DIR)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="/home/vivresavie/explainers-project/models/rf_adult.joblib")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--r2", type=float, default=0.9, help="R^2 threshold to accept explanation")
    p.add_argument("--kind", choices=["lime", "shap"], default="lime")
    p.add_argument("--shrink", type=float, default=0.5, help="Shrink factor per iteration (0-1]")
    p.add_argument("--kernel-width", type=float, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load artifact
    art = joblib.load(args.model)
    if "pipeline" not in art:
        raise RuntimeError("Model artifact must contain pipeline in joblib file (use train_model.py)")

    pipeline = art["pipeline"]
    X_test = art.get("X_test")
    if X_test is None:
        raise RuntimeError("Model artifact must contain X_test")

    # Create instance JSON and call XReason (assumes xreason runner at configured path)
    inst = X_test.iloc[args.index:args.index+1]
    inst_json = os.path.join(TMP_DIR, "instance_for_refine.json")
    save_json(series_to_jsonlike(inst.iloc[0]), inst_json)

    # call xreason to get region JSON
    xreason_out = os.path.join(TMP_DIR, "xreason_region_for_refine.json")
    try:
        xout = call_xreason_get_formal_region(args.model, inst_json, xreason_out, timeout=120)
    except Exception as e:
        raise RuntimeError(f"XReason runner failed: {e}")

    region = parse_xreason_region(xout)
    # Attach region to artifact and call refine loop
    art["initial_region"] = region

    print("Starting refinement loop for instance", args.index)
    res = refine_explanation_loop(
        artifact=art,
        instance_index=args.index,
        explainer_kind=args.kind,
        n_samples=args.samples,
        max_iters=args.max_iters,
        r2_threshold=args.r2,
        shrink_factor=args.shrink,
        kernel_width=args.kernel_width
    )

    # Save results
    out_path = os.path.join(TMP_DIR, f"refine_result_idx{args.index}_{args.kind}.json")
    save_json(res, out_path)
    print("Refinement complete. Results saved to", out_path)
    # Save optional CSV of the final samples used
    if res.get("samples") is not None:
        sf = res["samples"]
        if isinstance(sf, pd.DataFrame):
            sf.to_csv(os.path.join(TMP_DIR, f"refine_samples_idx{args.index}_{args.kind}.csv"), index=False)
            print("Saved final samples to CSV.")
