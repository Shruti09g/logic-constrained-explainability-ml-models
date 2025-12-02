import argparse
import os
import joblib
import pandas as pd
import time
import json
from tqdm import tqdm

from utils import ensure_dir, series_to_jsonlike
from xreason_interface import call_xreason_get_formal_region, parse_xreason_region
from region_sampler import sample_from_region, test_counterexamples
from explainers import Explainers

# Setup
TMP_DIR = "/home/vivresavie/explainers-project/temp_batch"
ensure_dir(TMP_DIR)

def run_batch(model_path, num_instances=20, samples=500, shap_k=3):
    # Load Model
    print(f"Loading model from {model_path}...")
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    X_test = artifact["X_test"]
    train_df = artifact["train_df"]
    feature_names = list(train_df.columns)
    
    expl = Explainers(pipeline, train_df, feature_names)
    
    results = []
    
    # Iterate
    print(f"Starting batch run on {num_instances} instances...")
    for i in tqdm(range(num_instances)):
        instance = X_test.iloc[i:i+1]
        
        # 1. Get Formal Region
        inst_json_path = os.path.join(TMP_DIR, f"inst_{i}.json")
        out_path = os.path.join(TMP_DIR, f"out_{i}.json")
        
        with open(inst_json_path, 'w') as f:
            json.dump(series_to_jsonlike(instance.iloc[0]), f)
            
        try:
            # Short timeout for batch to skip hard instances
            xout = call_xreason_get_formal_region(model_path, inst_json_path, out_path, timeout=30)
            if "region" not in xout:
                print(f"Skipping {i}: No region found.")
                continue
            region = parse_xreason_region(xout)
        except Exception:
            print(f"Skipping {i}: XReason timed out or failed.")
            continue

        # 2. Get SHAP Explanation
        try:
            orig_raw = series_to_jsonlike(instance.iloc[0])
            shap_top = expl.shap_explain(orig_raw, top_k=shap_k)
            shap_names = [n for n, _ in shap_top]
        except:
            continue
            
        # 3. Sampling Verification (Check for SHAP Counterexamples)
        # We use sampling here because running SAT on 100 instances takes too long.
        # Sampling is a "Lower Bound" on failure (if Sampling finds it, SAT definitely would).
        
        synth_samples = sample_from_region(region, instance.iloc[0], n_samples=samples)
        
        def model_predict(df): return pipeline.predict(df)
        
        def check_shap(df_row):
            try:
                row_dict = df_row.iloc[0].to_dict()
                # Check if top-k features match
                current_top = [n for n, _ in expl.shap_explain(row_dict, top_k=shap_k)]
                return set(current_top) == set(shap_names)
            except: return False

        cex_shap = test_counterexamples(synth_samples, model_predict, check_shap)
        
        # 4. Record Data
        results.append({
            "index": i,
            "prediction": int(instance.iloc[0].values[0]) if hasattr(instance.iloc[0], "values") else 0, # Placeholder
            "shap_failed": len(cex_shap) > 0,
            "shap_cex_count": len(cex_shap),
            "region_size": len(region) # simple complexity metric
        })
        
    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv("batch_results.csv", index=False)
    
    print("\n--- Batch Experiment Complete ---")
    print(f"Total Analyzed: {len(df_res)}")
    print(f"SHAP Failure Rate: {df_res['shap_failed'].mean():.2%}")
    print("Results saved to batch_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/vivresavie/explainers-project/models/rf_adult.joblib")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    
    run_batch(args.model, args.count)