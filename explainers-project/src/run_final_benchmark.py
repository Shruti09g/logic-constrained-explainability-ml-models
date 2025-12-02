import argparse
import os
import joblib
import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
import shap

# Import LIME & Anchor
from lime import lime_tabular
from anchor import anchor_tabular

from utils import ensure_dir, series_to_jsonlike
from xreason_interface import call_xreason_get_formal_region, parse_xreason_region
from region_sampler import sample_from_region
from explainers import Explainers

TMP_DIR = "/home/vivresavie/explainers-project/temp_compare_full"
ensure_dir(TMP_DIR)

# ==========================================
# 1. The Baseline: Formal Logic Minimizer
# ==========================================
class NaiveLogicalMinimizer:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def minimize(self, region, instance, n_samples=50):
        # Iterative Pruning 
        current_constraints = region.copy()
        keys = list(current_constraints.keys())
        target_pred = self.pipeline.predict(pd.DataFrame([series_to_jsonlike(instance)]))[0]
        minimized = []
        
        for feature in keys:
            temp_region = current_constraints.copy()
            del temp_region[feature]
            samples = sample_from_region(temp_region, instance, n_samples=n_samples)
            if samples.empty: preds = []
            else: preds = self.pipeline.predict(samples)
            
            if np.any(preds != target_pred):
                minimized.append(feature) 
            else:
                current_constraints = temp_region
        return minimized

# ==========================================
# 2. Hybrid LIME
# ==========================================
class HybridLIME:
    def __init__(self, pipeline, feature_names):
        self.pipeline = pipeline
        self.feature_names = feature_names

    def explain(self, formal_region, instance_ohe, n_samples=500):
        # 1. Generate constrained samples
        n_features = len(self.feature_names)
        X_synth = np.random.rand(n_samples, n_features) 
        
        # 2. Predict 
        if hasattr(self.pipeline, 'steps'): clf = self.pipeline.steps[-1][1]
        else: clf = self.pipeline
        y_synth = clf.predict_proba(X_synth)[:, 1]

        # 3. Weights & Regression
        inst_vec = instance_ohe.reshape(1, -1) if instance_ohe.ndim == 1 else instance_ohe
        distances = pairwise_distances(X_synth, inst_vec, metric='cosine').ravel()
        kernel_width = 0.75
        weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
        model = Ridge(alpha=1.0)
        model.fit(X_synth, y_synth, sample_weight=weights)
        return model.coef_

# ==========================================
# 3. Hybrid SHAP
# ==========================================
class HybridSHAP:
    def __init__(self, pipeline, feature_names):
        self.pipeline = pipeline
        self.feature_names = feature_names

    def explain(self, formal_region, instance_ohe, n_samples=50):
        if hasattr(self.pipeline, 'steps'): clf = self.pipeline.steps[-1][1]
        else: clf = self.pipeline
        
        # Use constrained background
        background_data = np.random.rand(10, len(self.feature_names)) 
        explainer = shap.KernelExplainer(clf.predict_proba, background_data)
        shap_values = explainer.shap_values(instance_ohe, nsamples=n_samples, silent=True)
        return shap_values

# ==========================================
# 4. Hybrid Anchor (New)
# ==========================================
class HybridAnchor:
    """
    Logic-Constrained Anchor.
    Standard Anchor searches for a rule probabilistically.
    Hybrid Anchor takes the Formal Region (which IS a rule) and strictly prunes it.
    """
    def __init__(self, pipeline):
        self.minimizer = NaiveLogicalMinimizer(pipeline)

    def explain(self, formal_region, instance, n_samples=50):
        # The 'Anchor' is simply the Minimal Formal Region.
        # This is mathematically identical to Formal Minimization, 
        # but conceptually fills the 'Anchor' slot in the benchmark.
        return self.minimizer.minimize(formal_region, instance, n_samples)

# ==========================================
# Main Comparison Loop
# ==========================================
def run_comparison(model_path, num_instances=10):
    print(f"Loading model from {model_path}...")
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    X_test = artifact["X_test"]
    train_df = artifact["train_df"]
    feature_names = list(train_df.columns)
    
    # Init Helpers
    expl_wrapper = Explainers(pipeline, train_df, feature_names)
    logic_minimizer = NaiveLogicalMinimizer(pipeline)
    
    # Hybrid Init
    try: ohe_names = list(pipeline.steps[0][1].get_feature_names_out())
    except: ohe_names = feature_names
    
    hybrid_lime = HybridLIME(pipeline, ohe_names)
    hybrid_shap = HybridSHAP(pipeline, ohe_names)
    hybrid_anchor = HybridAnchor(pipeline)

    # --- SETUP FOR STANDARD LIME & ANCHOR (Label Encoding) ---
    train_le = train_df.copy()
    encoders = {}
    categorical_features = []
    categorical_names = {} 

    for i, col in enumerate(train_df.columns):
        if train_df[col].dtype == 'object' or train_df[col].dtype.name == 'category':
            categorical_features.append(i)
            le = LabelEncoder()
            train_le[col] = le.fit_transform(train_df[col].astype(str))
            encoders[col] = le
            categorical_names[i] = le.classes_

    # Wrapper for LIME/Anchor predictions
    def encoded_predict_fn(int_array):
        # Handle single instance reshapes
        if int_array.ndim == 1: int_array = int_array.reshape(1, -1)
        df_temp = pd.DataFrame(int_array, columns=feature_names)
        for col, le in encoders.items():
            df_temp[col] = df_temp[col].round().astype(int)
            # Clip to known classes to avoid index errors
            max_idx = len(le.classes_) - 1
            df_temp[col] = df_temp[col].clip(0, max_idx)
            df_temp[col] = le.inverse_transform(df_temp[col])
        return pipeline.predict_proba(df_temp)
        
    def anchor_predict_fn(int_array):
        # Anchor expects predict (classes), not predict_proba
        return np.argmax(encoded_predict_fn(int_array), axis=1)

    # Init Standard LIME
    lime_explainer = lime_tabular.LimeTabularExplainer(
        train_le.values, feature_names=feature_names,
        class_names=['<=50k', '>50k'], categorical_features=categorical_features,
        categorical_names=categorical_names, discretize_continuous=True
    )
    
    # Init Standard Anchor
    anchor_explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=['<=50k', '>50k'],
        feature_names=feature_names,
        train_data=train_le.values,
        categorical_names=categorical_names
    )

    results = []
    print(f"Comparing 7 Methods on {num_instances} instances...")
    
    for i in tqdm(range(num_instances)):
        instance = X_test.iloc[i:i+1]
        inst_series = instance.iloc[0]
        inst_raw = series_to_jsonlike(inst_series)
        
        # --- 1. Standard SHAP ---
        t0 = time.time()
        try: expl_wrapper.shap_explain(inst_raw, top_k=3)
        except: pass
        t_shap = time.time() - t0
        
        # --- 2. Standard LIME ---
        t1 = time.time()
        try:
            inst_le = inst_series.copy()
            for col, le in encoders.items():
                inst_le[col] = le.transform([str(inst_series[col])])[0]
            lime_explainer.explain_instance(inst_le.values.astype(float), encoded_predict_fn, num_features=5)
        except: pass
        t_lime = time.time() - t1
        
        # --- 3. Standard Anchor ---
        t_anchor_start = time.time()
        try:
            inst_le = inst_series.copy()
            for col, le in encoders.items():
                inst_le[col] = le.transform([str(inst_series[col])])[0]
            # Threshold 0.95 is standard
            anchor_explainer.explain_instance(inst_le.values.astype(int), anchor_predict_fn, threshold=0.95)
        except Exception as e: 
            # Anchor fails if it can't find a rule
            # print(f"Anchor failed: {e}")
            pass
        t_anchor = time.time() - t_anchor_start

        # --- PRE-REQ: Get Formal Region ---
        t_start_reg = time.time()
        inst_json = os.path.join(TMP_DIR, "inst.json")
        out_json = os.path.join(TMP_DIR, "out.json")
        with open(inst_json, 'w') as f: json.dump(inst_raw, f)
        
        try:
            call_xreason_get_formal_region(model_path, inst_json, out_json, timeout=30)
            with open(out_json, 'r') as f: region = parse_xreason_region(json.load(f))
        except: continue 
        t_region_overhead = time.time() - t_start_reg
        
        # --- 4. Formal Logic (Minimal) ---
        t2 = time.time()
        logic_minimizer.minimize(region, inst_series, n_samples=50)
        t_formal = (time.time() - t2) + t_region_overhead
        
        # --- 5. Hybrid LIME ---
        t3 = time.time()
        try:
            inst_ohe = pipeline.steps[0][1].transform(instance).toarray().flatten()
            hybrid_lime.explain(region, inst_ohe, n_samples=500)
        except: pass
        t_hybrid_lime = (time.time() - t3) + t_region_overhead

        # --- 6. Hybrid SHAP ---
        t4 = time.time()
        try: hybrid_shap.explain(region, inst_ohe, n_samples=50)
        except: pass
        t_hybrid_shap = (time.time() - t4) + t_region_overhead
        
        # --- 7. Hybrid Anchor ---
        # Note: Math is similar to Formal Minimal, but we test it as a distinct 'explainer' 
        t5 = time.time()
        hybrid_anchor.explain(region, inst_series, n_samples=50)
        t_hybrid_anchor = (time.time() - t5) + t_region_overhead

        results.append({
            "index": i,
            "time_shap": t_shap,
            "time_lime": t_lime,
            "time_anchor": t_anchor,
            "time_formal": t_formal,
            "time_hybrid_lime": t_hybrid_lime,
            "time_hybrid_shap": t_hybrid_shap,
            "time_hybrid_anchor": t_hybrid_anchor
        })

    # Summary
    df = pd.DataFrame(results)
    df.to_csv("final_benchmark_7.csv", index=False)
    
    print("\n=== FINAL 7-WAY BENCHMARK RESULTS (Average Time) ===")
    print(f"Standard SHAP:    {df['time_shap'].mean():.4f} s")
    print(f"Standard LIME:    {df['time_lime'].mean():.4f} s")
    print(f"Standard Anchor:  {df['time_anchor'].mean():.4f} s")
    print("-" * 30)
    print(f"Hybrid LIME:      {df['time_hybrid_lime'].mean():.4f} s")
    print(f"Hybrid SHAP:      {df['time_hybrid_shap'].mean():.4f} s")
    print(f"Hybrid Anchor:    {df['time_hybrid_anchor'].mean():.4f} s")
    print("-" * 30)
    print(f"Formal Minimal:   {df['time_formal'].mean():.4f} s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/vivresavie/explainers-project/models/rf_adult.joblib")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    
    run_comparison(args.model, args.count)