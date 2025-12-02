"""
Run demo: XReason region extraction + LIME & SHAP explanations + Hybrid Logic-Constrained Explanation
+ Sampling & SAT-guided counterexample search.

Usage (examples):
  python3 src/run_demo.py --model models/rf_adult.joblib --index 0 --samples 500
  python3 src/run_demo.py --model models/rf_adult.joblib --index 0 --samples 500 --use-sat --sat-bins 12 --sat-tries 2000
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from typing import Optional

from utils import ensure_dir, save_json, series_to_jsonlike
from xreason_interface import call_xreason_get_formal_region, parse_xreason_region
from region_sampler import sample_from_region, test_counterexamples
from explainers import Explainers

# SAT-guided module (optional; requires python-sat)
try:
    from sat_counterexample import find_counterexample_with_sat
    SAT_AVAILABLE = True
except Exception:
    find_counterexample_with_sat = None
    SAT_AVAILABLE = False

# Temporary output dir
TMP_DIR = "/home/vivresavie/explainers-project/temp"
ensure_dir(TMP_DIR)

# ==========================================
# NEW: Adaptive Logic-Constrained Explainer
# ==========================================
class LogicConstrainedExplainer:
    def __init__(self, model_pipeline, feature_names):
        self.model = model_pipeline
        self.feature_names = feature_names

    def generate_constrained_samples(self, formal_region, original_instance, n_samples=1000, expansion=0.0):
        """
        Generates samples. If expansion > 0, expands numeric bounds by that %.
        """
        n_features = len(self.feature_names)
        samples = np.zeros((n_samples, n_features))
        
        if hasattr(original_instance, "values"):
            orig_vals = original_instance.values.flatten()
        else:
            orig_vals = original_instance

        for col_idx, col_name in enumerate(self.feature_names):
            # Parse column name
            clean_col = col_name
            if "__" in clean_col: clean_col = clean_col.split("__", 1)[1]
            if '_' in clean_col: base_feat, cat_val = clean_col.rsplit('_', 1)
            else: base_feat, cat_val = clean_col, None

            # Apply Constraints
            if base_feat in formal_region:
                constraint = formal_region[base_feat]
                
                # Case A: Categorical (Fixed - never expand)
                if constraint[0] == 'eq':
                    required_val = constraint[1]
                    if str(cat_val) == str(required_val):
                        samples[:, col_idx] = 1.0
                    else:
                        samples[:, col_idx] = 0.0
                
                # Case B: Numeric Interval (Apply expansion here)
                else:
                    low, high = constraint
                    # Check for safety
                    if low == -float('inf'): low = -1e9
                    if high == float('inf'): high = 1e9
                    
                    # Expansion logic: widen the window to find sensitivity
                    span = high - low
                    if span == 0: span = 1 # avoid zero div
                    margin = span * expansion
                    
                    # Sample Uniformly
                    samples[:, col_idx] = np.random.uniform(low - margin, high + margin, n_samples)
            
            else:
                # Case C: Unconstrained -> Fix to original
                samples[:, col_idx] = orig_vals[col_idx]

        return samples

    def explain(self, formal_region, original_instance, n_samples=1000):
        # Phase 1: Strict Check
        explanation = self._run_regression(formal_region, original_instance, n_samples, expansion=0.0)
        
        # If strict check shows 0 influence (Flat region), RELAX constraints to find nearest boundary
        if not explanation:
            print("  -> Region is perfectly stable (Flat). Expanding bounds by 20% to find boundary sensitivity...")
            explanation = self._run_regression(formal_region, original_instance, n_samples, expansion=0.2)
            
        return explanation

    def _run_regression(self, formal_region, original_instance, n_samples, expansion):
        # 1. Generate
        X_synth = self.generate_constrained_samples(formal_region, original_instance, n_samples, expansion)
        
        # 2. Predict (Bypass Preprocessor)
        if hasattr(self.model, 'steps'):
            classifier = self.model.steps[-1][1]
        else:
            classifier = self.model
        y_synth = classifier.predict_proba(X_synth)[:, 1]
        
        # Check variance
        if np.var(y_synth) < 1e-6:
            return [] # No variance, coefficients will be meaningless

        # 3. Train
        surrogate = LinearRegression()
        surrogate.fit(X_synth, y_synth)
        
        # 4. Extract
        coeffs = list(zip(self.feature_names, surrogate.coef_))
        explanation = [(f, w) for f, w in coeffs if abs(w) > 0.0001]
        explanation.sort(key=lambda x: abs(x[1]), reverse=True)
        return explanation

# ==========================================
# Helper Functions
# ==========================================

def pretty_print_region(region: dict):
    print("Parsed formal region:")
    for k, v in region.items():
        print(f"  {k}: {v}")

def diagnose_shap_failure(original_row, cex_row, shap_features):
    """
    Forensic analysis: Did the features SHAP claimed were important actually change?
    """
    print(f"\n[DIAGNOSTICS] Analyzing SHAP Failure...")
    print(f"SHAP claimed top features: {shap_features}")
    
    suspects = []
    
    # original_row is a Series with raw values
    # cex_row is likely a Series or dict with raw values
    
    for feat in shap_features:
        # SHAP names might be transformed (e.g. ohe__country_Germany)
        # We need to map back to raw feature to check value change, 
        # OR check the transformed value if CEX provides it.
        # Simple check: does the string exist in the keys?
        
        clean_feat = feat
        if "__" in feat: clean_feat = feat.split("__", 1)[1]
        if "_" in clean_feat: clean_feat = clean_feat.rsplit("_", 1)[0] # Extract 'native-country'
        
        if clean_feat in original_row.index:
            orig_val = original_row[clean_feat]
            cex_val = cex_row[clean_feat] if clean_feat in cex_row else "Unknown"
            
            if str(orig_val) == str(cex_val):
                suspects.append(feat)
                print(f"  -> WARNING: {clean_feat} did NOT change (Val: {orig_val}), yet prediction flipped.")
            else:
                print(f"  -> {clean_feat} changed from {orig_val} to {cex_val}.")
        else:
            # Fallback for complex names
            print(f"  -> Could not map {feat} to raw data for comparison.")

    if suspects:
        print(f"CONCLUSION: SHAP likely hallucinated importance for {suspects}.")
    else:
        print("CONCLUSION: SHAP features were involved in the flip.")

def safe_lime_top1(explainer: Explainers, row):
    try:
        exp = explainer.lime_explain(row, num_features=1)
        lst = exp.as_list()
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0][0]
    except Exception:
        pass
    return None

def safe_shap_topk(explainer: Explainers, row, k: int):
    try:
        vals = explainer.shap_explain(row, top_k=k)
        names = [n for n, _ in vals]
        return names
    except Exception:
        return []

# ==========================================
# Main Pipeline
# ==========================================

def run_one_instance(
    model_artifact_path: str,
    instance_index: int = 0,
    n_samples: int = 500,
    expand_numeric: float = 0.0,
    shap_top_k: int = 3,
    use_sat: bool = False,
    sat_bins: int = 8,
    sat_tries: int = 1000,
    sat_solver: str = "glucose3",
    sat_verbose: bool = False
):
    ensure_dir(TMP_DIR)

    # --- Load artifact ---
    artifact = joblib.load(model_artifact_path)
    pipeline = artifact["pipeline"]
    X_test = artifact["X_test"]
    train_df = artifact["train_df"]

    if instance_index >= len(X_test):
        raise IndexError("instance_index out of range for X_test")

    instance = X_test.iloc[instance_index:instance_index + 1]
    inst_json_path = os.path.join(TMP_DIR, "instance.json")
    save_json(series_to_jsonlike(instance.iloc[0]), inst_json_path)

    # --- Call XReason ---
    xreason_out_path = os.path.join(TMP_DIR, "xreason_out.json")
    try:
        xout = call_xreason_get_formal_region(model_artifact_path, inst_json_path, xreason_out_path, timeout=120)
    except Exception as e:
        print("Error calling XReason:", e)
        return

    region = parse_xreason_region(xout)
    pretty_print_region(region)
    save_json(xout, xreason_out_path)

    # --- Initialize explainers ---
    raw_feature_names = list(train_df.columns)
    
    # Attempt to get OHE feature names for Hybrid
    try:
        preprocessor = pipeline.steps[0][1]
        ohe_feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        print("Warning: Could not extract OHE feature names. Hybrid explainer might fail.")
        ohe_feature_names = []

    expl = Explainers(pipeline, train_df, raw_feature_names)

    # --- 1. Standard Explanations ---
    orig_raw = series_to_jsonlike(instance.iloc[0])

    # LIME
    try:
        lime_exp = expl.lime_explain(orig_raw, num_features=5)
        lime_list = lime_exp.as_list()
    except Exception as e:
        lime_list = str(e)
    print("\nLIME explanation (top features):")
    print(lime_list)
    lime_top1 = lime_list[0][0] if isinstance(lime_list, list) and len(lime_list) > 0 else None

    # SHAP
    try:
        shap_top = expl.shap_explain(orig_raw, top_k=shap_top_k)
        print("\nSHAP top features (transformed names and values):")
        for name, val in shap_top:
            v = float(val) if isinstance(val, (int, float)) else float(pd.np.array(val).flatten()[0])
            print(f"  {name}: {v:.6f}")
        shap_top_names = [name for name, _ in shap_top]
    except Exception as e:
        print("\nSHAP failed:", e)
        shap_top = []
        shap_top_names = []

    # --- 2. NEW: Hybrid (Logic-Constrained) Explanation ---
    print("\n--- Running Logic-Constrained Explainer (Hybrid) ---")
    if len(ohe_feature_names) > 0:
        try:
            # A. Transform the raw instance to OHE (get the 0s and 1s)
            preprocessor = pipeline.steps[0][1] 
            instance_ohe = preprocessor.transform(instance)
            
            # Ensure it's a flat array
            if hasattr(instance_ohe, "toarray"):
                instance_ohe = instance_ohe.toarray()
            instance_ohe = instance_ohe.flatten()

            # B. Run the Explainer
            hybrid_explainer = LogicConstrainedExplainer(pipeline, ohe_feature_names)
            hybrid_exp = hybrid_explainer.explain(region, instance_ohe, n_samples=1000)
            
            print("Hybrid Top Features (Logic-Guided):")
            for feat, weight in hybrid_exp[:5]:
                print(f"  {feat}: {weight:.4f}")
        except Exception as e:
            print(f"Hybrid explainer failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping Hybrid: No OHE feature names found.")


    # --- 3. Sampling Verification ---
    samples = sample_from_region(region, instance.iloc[0], n_samples=n_samples, random_state=42, expand_numeric=expand_numeric)
    print(f"\nGenerated {len(samples)} samples inside the formal region.")

    def model_predict_fn(df: pd.DataFrame):
        return pipeline.predict(df)

    # Heuristic assertions
    def heuristic_assertion_lime(df_row):
        try:
            row_dict = df_row.to_dict() if isinstance(df_row, pd.Series) else df_row.iloc[0].to_dict()
            top = safe_lime_top1(expl, row_dict)
            return top == lime_top1
        except: return False

    def heuristic_assertion_shap(df_row):
        try:
            row_dict = df_row.to_dict() if isinstance(df_row, pd.Series) else df_row.iloc[0].to_dict()
            names = safe_shap_topk(expl, row_dict, shap_top_k)
            return set(names) == set(shap_top_names)
        except: return False

    print("\nTesting counterexamples for LIME (sampling)...")
    cex_lime = test_counterexamples(samples, model_predict_fn, lambda r: heuristic_assertion_lime(r))
    print("LIME counterexamples found:", len(cex_lime))
    
    print("\nTesting counterexamples for SHAP (sampling)...")
    cex_shap = test_counterexamples(samples, model_predict_fn, lambda r: heuristic_assertion_shap(r))
    print("SHAP counterexamples found:", len(cex_shap))

    if not cex_shap.empty:
        cex_shap.to_csv(os.path.join(TMP_DIR, "counterexamples_shap.csv"), index=False)

    # --- 4. SAT-Guided Verification ---
    sat_cex_lime = None
    sat_cex_shap = None
    
    if use_sat and SAT_AVAILABLE:
        print("\nLaunching SAT-guided search for SHAP counterexample...")
        sat_heu_shap = lambda series_row: heuristic_assertion_shap(series_row)
        
        sat_cex = find_counterexample_with_sat(
            region=region,
            original_instance=instance.iloc[0],
            model_predict_fn=model_predict_fn,
            heuristic_assertion_fn=sat_heu_shap,
            num_bins=sat_bins,
            max_tries=sat_tries,
            solver_name=sat_solver,
            verbose=sat_verbose
        )
        
        if sat_cex is not None:
            sat_cex_shap = sat_cex
            pd.DataFrame([sat_cex]).to_csv(os.path.join(TMP_DIR, "cex_shap_sat.csv"), index=False)
            print("SAT found SHAP counterexample -> temp/cex_shap_sat.csv")
            
            # Run Diagnostics
            diagnose_shap_failure(instance.iloc[0], pd.Series(sat_cex), shap_top_names)
        else:
            print("SAT search found no SHAP counterexample.")

    # --- Save Meta ---
    meta = {
        "instance_index": int(instance_index),
        "n_samples": n_samples,
        "n_counterexamples_shap_sampling": int(len(cex_shap)),
        "sat_found_shap": sat_cex_shap is not None
    }
    save_json(meta, os.path.join(TMP_DIR, "run_meta.json"))
    print("\nRun complete.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/vivresavie/explainers-project/models/rf_adult.joblib")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--expand", type=float, default=0.0)
    parser.add_argument("--shap-k", type=int, default=3)
    parser.add_argument("--use-sat", action="store_true")
    parser.add_argument("--sat-bins", type=int, default=8)
    parser.add_argument("--sat-tries", type=int, default=1000)
    parser.add_argument("--sat-solver", type=str, default="glucose3")
    parser.add_argument("--sat-verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_one_instance(
        model_artifact_path=args.model,
        instance_index=args.index,
        n_samples=args.samples,
        expand_numeric=args.expand,
        shap_top_k=args.shap_k,
        use_sat=args.use_sat,
        sat_bins=args.sat_bins,
        sat_tries=args.sat_tries,
        sat_solver=args.sat_solver,
        sat_verbose=args.sat_verbose
    )