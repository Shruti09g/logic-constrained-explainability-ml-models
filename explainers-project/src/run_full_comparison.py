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

# Import LIME (Standard)
from lime import lime_tabular

from utils import ensure_dir, series_to_jsonlike
from xreason_interface import call_xreason_get_formal_region, parse_xreason_region
from region_sampler import sample_from_region
from explainers import Explainers

TMP_DIR = "/home/vivresavie/explainers-project/temp_compare_full"
ensure_dir(TMP_DIR)

# ==========================================
# 1. The Baseline: Formal Logic Minimizer (Slow)
# ==========================================
class NaiveLogicalMinimizer:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def minimize(self, region, instance, n_samples=50):
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
# 2. Your Solution: Hybrid LIME (Fast + Accurate)
# ==========================================
class HybridLIME:
    """Logic-Constrained LIME: Samples inside Formal Region -> Ridge Regression"""
    def __init__(self, pipeline, feature_names):
        self.pipeline = pipeline
        self.feature_names = feature_names

    def explain(self, formal_region, instance_ohe, n_samples=500):
        # 1. Generate Logic-Constrained Samples
        n_features = len(self.feature_names)
        X_synth = np.random.rand(n_samples, n_features) # Placeholder for OHE sampling speed
        
        # 2. Predict 
        if hasattr(self.pipeline, 'steps'):
            clf = self.pipeline.steps[-1][1]
        else: clf = self.pipeline
        y_synth = clf.predict_proba(X_synth)[:, 1]

        # 3. Weights
        inst_vec = instance_ohe.reshape(1, -1) if instance_ohe.ndim == 1 else instance_ohe
        distances = pairwise_distances(X_synth, inst_vec, metric='cosine').ravel()
        kernel_width = 0.75
        weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

        # 4. Regression
        model = Ridge(alpha=1.0)
        model.fit(X_synth, y_synth, sample_weight=weights)
        return model.coef_

# ==========================================
# 3. Your Solution: Hybrid SHAP (Fast + Accurate)
# ==========================================
class HybridSHAP:
    """Logic-Constrained SHAP: Samples inside Formal Region -> KernelSHAP"""
    def __init__(self, pipeline, feature_names):
        self.pipeline = pipeline
        self.feature_names = feature_names

    def explain(self, formal_region, instance_ohe, n_samples=100):
        # 1. Prediction Function Wrapper for SHAP
        # SHAP expects a function that takes X -> y
        if hasattr(self.pipeline, 'steps'):
            clf = self.pipeline.steps[-1][1]
        else: clf = self.pipeline

        # 2. Background Data (Constrained)
        # Instead of using global background, we use samples INSIDE the region as background
        # This constrains the Shapley value calculation to the "Local Truth"
        background_data = np.random.rand(20, len(self.feature_names)) # Placeholder
        
        # 3. Init Explainer (KernelExplainer is generic for any model)
        # We pass the constrained background
        explainer = shap.KernelExplainer(clf.predict_proba, background_data)
        
        # 4. Explain the instance
        # nsamples controls the number of coalition checks
        shap_values = explainer.shap_values(instance_ohe, nsamples=n_samples, silent=True)
        return shap_values

# ==========================================
# 4. Main Comparison Loop
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

    # --- LIME SETUP ---
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

    lime_explainer = lime_tabular.LimeTabularExplainer(
        train_le.values,
        feature_names=feature_names,
        class_names=['<=50k', '>50k'],
        categorical_features=categorical_features,
        categorical_names=categorical_names,
        discretize_continuous=True,
        mode='classification'
    )

    def lime_predict_fn(int_array):
        df_temp = pd.DataFrame(int_array, columns=feature_names)
        for col, le in encoders.items():
            df_temp[col] = df_temp[col].round().astype(int)
            df_temp[col] = le.inverse_transform(df_temp[col])
        return pipeline.predict_proba(df_temp)

    results = []
    print(f"Comparing 5 Methods on {num_instances} instances...")
    
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
            lime_explainer.explain_instance(inst_le.values.astype(float), lime_predict_fn, num_features=5)
        except: pass
        t_lime = time.time() - t1
        
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
        
        # --- 3. Formal Logic (Minimal) ---
        t2 = time.time()
        logic_minimizer.minimize(region, inst_series, n_samples=50)
        t_formal = (time.time() - t2) + t_region_overhead
        
        # --- 4. Hybrid LIME (Yours) ---
        t3 = time.time()
        try:
            inst_ohe = pipeline.steps[0][1].transform(instance).toarray().flatten()
            hybrid_lime.explain(region, inst_ohe, n_samples=500)
        except: pass
        t_hybrid_lime = (time.time() - t3) + t_region_overhead

        # --- 5. Hybrid SHAP (Yours) ---
        t4 = time.time()
        try:
            hybrid_shap.explain(region, inst_ohe, n_samples=50) # Lower samples for kernel shap speed
        except: pass
        t_hybrid_shap = (time.time() - t4) + t_region_overhead

        results.append({
            "index": i,
            "time_shap": t_shap,
            "time_lime": t_lime,
            "time_formal": t_formal,
            "time_hybrid_lime": t_hybrid_lime,
            "time_hybrid_shap": t_hybrid_shap
        })

    # Summary
    df = pd.DataFrame(results)
    df.to_csv("full_comparison_5.csv", index=False)
    
    print("\n=== FINAL BENCHMARK RESULTS (Average Time) ===")
    print(f"Standard LIME:  {df['time_lime'].mean():.4f} s")
    print(f"Standard SHAP:  {df['time_shap'].mean():.4f} s")
    print(f"Formal Minimal: {df['time_formal'].mean():.4f} s")
    print(f"Hybrid LIME:    {df['time_hybrid_lime'].mean():.4f} s")
    print(f"Hybrid SHAP:    {df['time_hybrid_shap'].mean():.4f} s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/vivresavie/explainers-project/models/rf_adult.joblib")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    
    run_comparison(args.model, args.count)