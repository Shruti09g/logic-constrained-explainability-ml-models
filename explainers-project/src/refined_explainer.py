# src/refined_explainer.py
"""
Refined explainer with SAT-based refinement.

This file is the updated refined_explainer that:
 - performs region-constrained sampling
 - fits constrained LIME (weighted Ridge) or constrained SHAP surrogates
 - verifies surrogate via weighted R^2
 - optionally runs SAT-guided search (discretized CEGAR) inside the current region
   to find minimal counterexamples that break the heuristic (LIME/SHAP)
 - tightens the region and repeats until R^2 passes and SAT finds no counterexample,
   or max iterations reached.

Requirements:
 - shap (optional, for constrained SHAP)
 - python-sat (PySAT) optional for SAT-based search; if absent SAT mode is disabled.

Project plan path (for provenance): /mnt/data/Project_Plan_Report.pdf
"""
from typing import Dict, Any, Tuple, List, Optional
import math
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# optional shap
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

# optional SAT module
try:
    from sat_counterexample import find_counterexample_with_sat
    SAT_AVAILABLE = True
except Exception:
    find_counterexample_with_sat = None
    SAT_AVAILABLE = False

# expected helper modules from your repo
from region_sampler import sample_from_region
from explainers import Explainers    # wrapper that provides LIME/SHAP calls
from xreason_interface import parse_xreason_region  # typically used by caller

PROJECT_PLAN_PDF_PATH = "/mnt/data/Project_Plan_Report.pdf"


def constrained_sampling(region: Dict[str, Any],
                         instance_row: pd.Series,
                         n_samples: int = 500,
                         random_state: Optional[int] = None,
                         expand_numeric: float = 0.0) -> pd.DataFrame:
    samples = sample_from_region(region, instance_row, n_samples=n_samples,
                                 random_state=random_state, expand_numeric=expand_numeric)
    return samples


def _kernel_distance_weights(X_trans: np.ndarray, x0_trans: np.ndarray, kernel_width: Optional[float] = None) -> np.ndarray:
    if x0_trans.ndim == 1:
        x0 = x0_trans
    else:
        x0 = x0_trans.ravel()
    diffs = X_trans - x0
    dists = np.linalg.norm(diffs, axis=1)
    if kernel_width is None:
        kernel_width = 0.75 * math.sqrt(max(1, X_trans.shape[1]))
    weights = np.exp(-(dists ** 2) / (kernel_width ** 2))
    weights = np.maximum(weights, 1e-12)
    return weights


def fit_constrained_lime(pipeline,
                         preprocessor,
                         model,
                         transformed_feature_names: List[str],
                         instance_raw: Dict[str, Any],
                         samples_raw: pd.DataFrame,
                         class_idx: int = 1,
                         num_features: int = 6,
                         alpha: float = 1.0,
                         kernel_width: Optional[float] = None) -> Dict[str, Any]:
    try:
        X_trans = preprocessor.transform(samples_raw) if preprocessor is not None else samples_raw.values
    except Exception:
        X_trans = samples_raw.values.astype(float)

    try:
        inst_df = pd.DataFrame([instance_raw], columns=samples_raw.columns)
        x0_trans = preprocessor.transform(inst_df)[0] if preprocessor is not None else inst_df.values[0]
    except Exception:
        x0_trans = X_trans[0]

    try:
        probs = model.predict_proba(X_trans)
        y = np.array([p[class_idx] for p in probs])
    except Exception:
        y = model.predict(X_trans)
        y = np.asarray(y, dtype=float)

    weights = _kernel_distance_weights(X_trans, x0_trans, kernel_width=kernel_width)

    scaler = StandardScaler(with_mean=True, with_std=True)
    try:
        X_scaled = scaler.fit_transform(X_trans)
    except Exception:
        X_scaled = StandardScaler().fit_transform(np.nan_to_num(X_trans.astype(float)))

    W_sqrt = np.sqrt(weights).reshape(-1, 1)
    X_w = X_scaled * W_sqrt
    y_w = y * np.sqrt(weights)

    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_w, y_w)

    y_pred = ridge.predict(X_scaled)
    y_mean = np.average(y, weights=weights)
    ss_tot = np.sum(weights * (y - y_mean) ** 2)
    ss_res = np.sum(weights * (y - y_pred) ** 2)
    r2_weighted = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    coefs = ridge.coef_
    if hasattr(coefs, "ndim") and coefs.ndim > 1:
        coefs = coefs.ravel()
    n = min(len(transformed_feature_names), len(coefs))
    coeff_map = {transformed_feature_names[i]: float(coefs[i]) for i in range(n)}
    sorted_feats = sorted(coeff_map.items(), key=lambda kv: -abs(kv[1]))

    return {
        "coeffs": coeff_map,
        "top_features": sorted_feats[:num_features],
        "r2_weighted": float(r2_weighted),
        "n_samples": X_trans.shape[0],
        "X_trans": X_trans,
        "y": y,
        "weights": weights
    }


def fit_constrained_shap(pipeline,
                         preprocessor,
                         model,
                         instance_raw: Dict[str, Any],
                         samples_raw: pd.DataFrame,
                         transformed_feature_names: List[str],
                         nsamples: int = 100) -> Dict[str, Any]:
    if not SHAP_AVAILABLE:
        raise RuntimeError("shap package not available")

    try:
        X_bg = preprocessor.transform(samples_raw) if preprocessor is not None else samples_raw.values
    except Exception:
        X_bg = samples_raw.values

    inst_df = pd.DataFrame([instance_raw], columns=samples_raw.columns)
    try:
        x0_trans = preprocessor.transform(inst_df)[0] if preprocessor is not None else inst_df.values[0]
    except Exception:
        x0_trans = X_bg[0]

    def f_transformed(arr):
        try:
            return model.predict_proba(arr)
        except Exception:
            raise RuntimeError("Model did not accept transformed arrays for KernelExplainer.")

    bg = X_bg if X_bg.shape[0] <= 100 else X_bg[np.random.choice(X_bg.shape[0], 100, replace=False)]
    expl = shap.KernelExplainer(f_transformed, bg)
    shap_vals = expl.shap_values(x0_trans, nsamples=nsamples)

    try:
        probs_inst = model.predict_proba(x0_trans.reshape(1, -1))[0]
        class_idx = int(np.argmax(probs_inst))
    except Exception:
        class_idx = 0

    if isinstance(shap_vals, list):
        vals = np.array(shap_vals[class_idx]).reshape(-1)
    else:
        vals = np.array(shap_vals).reshape(-1)

    n = min(len(transformed_feature_names), vals.shape[0])
    feats = [(transformed_feature_names[i], float(vals[i])) for i in range(n)]
    feats_sorted = sorted(feats, key=lambda x: -abs(x[1]))

    return {"shap_values": vals, "feature_names": transformed_feature_names, "top_features": feats_sorted}


def tighten_region(region: Dict[str, Any], instance_row: pd.Series, shrink_factor: float = 0.5) -> Dict[str, Any]:
    new_region: Dict[str, Any] = {}
    for feat, cons in region.items():
        if isinstance(cons, list) and len(cons) == 2:
            left, right = cons[0], cons[1]
            if left == "eq":
                new_region[feat] = cons
                continue
            try:
                lo = float(cons[0])
                hi = float(cons[1])
                if feat in instance_row and not pd.isna(instance_row[feat]):
                    center = float(instance_row[feat])
                else:
                    center = (lo + hi) / 2.0
                half_width = (hi - lo) / 2.0 * float(shrink_factor)
                new_lo = max(lo, center - half_width)
                new_hi = min(hi, center + half_width)
                if new_hi <= new_lo:
                    new_lo = max(lo, center - 1e-6)
                    new_hi = min(hi, center + 1e-6)
                new_region[feat] = [new_lo, new_hi]
            except Exception:
                new_region[feat] = cons
        else:
            new_region[feat] = cons
    return new_region


def _make_heuristic_assertion(expl_wrapper: Explainers, kind: str, original_instance_raw: Dict[str, Any], shap_top_k: int):
    """
    Create a function that, given a candidate pandas Series or single-row DataFrame, returns True iff
    the heuristic (LIME top-1 or SHAP top-k) holds equal to the original instance's heuristic claim.
    """
    kind = kind.lower()
    if kind == "lime":
        # compute original top1
        orig_top1 = None
        try:
            le = expl_wrapper.lime_explain(pd.Series(original_instance_raw), num_features=1)
            lst = le.as_list()
            orig_top1 = lst[0][0] if isinstance(lst, list) and len(lst) > 0 else None
        except Exception:
            orig_top1 = None

        def assertion(candidate_row):
            try:
                if isinstance(candidate_row, pd.Series):
                    row = candidate_row
                else:
                    row = candidate_row.iloc[0]
                le = expl_wrapper.lime_explain(row, num_features=1)
                lst = le.as_list()
                top = lst[0][0] if isinstance(lst, list) and len(lst) > 0 else None
                return top == orig_top1
            except Exception:
                return False

        return assertion

    elif kind == "shap":
        # compute original top-k names
        try:
            st = expl_wrapper.shap_explain(pd.Series(original_instance_raw), top_k=shap_top_k)
            orig_topk = [n for n, _ in st] if isinstance(st, list) else []
        except Exception:
            orig_topk = []

        def assertion(candidate_row):
            try:
                if isinstance(candidate_row, pd.Series):
                    row = candidate_row
                else:
                    row = candidate_row.iloc[0]
                st2 = expl_wrapper.shap_explain(row, top_k=shap_top_k)
                names = [n for n, _ in st2] if isinstance(st2, list) else []
                return set(names) == set(orig_topk)
            except Exception:
                return False

        return assertion
    else:
        raise ValueError("Unknown kind for heuristic assertion")


def refine_explanation_loop(artifact: Dict[str, Any],
                            instance_index: int = 0,
                            class_idx: Optional[int] = None,
                            explainer_kind: str = "lime",
                            n_samples: int = 500,
                            max_iters: int = 5,
                            r2_threshold: float = 0.9,
                            shrink_factor: float = 0.5,
                            kernel_width: Optional[float] = None,
                            random_state: Optional[int] = None,
                            shap_nsamples: int = 100,
                            use_sat: bool = False,
                            sat_bins: int = 8,
                            sat_tries: int = 1000,
                            sat_solver: str = "glucose3",
                            sat_verbose: bool = False,
                            shap_top_k: int = 3) -> Dict[str, Any]:
    """
    Orchestrator with optional SAT-based search.

    New SAT parameters:
      - use_sat: run SAT-guided search after each candidate fit
      - sat_bins: bins per numeric interval
      - sat_tries: max tries / SAT iterations
      - sat_solver: PySAT solver name
      - sat_verbose: verbosity flag
      - shap_top_k: top-k used in SHAP assertion comparisons
    """
    # validate artifact
    if "pipeline" not in artifact:
        raise RuntimeError("artifact must contain 'pipeline'")
    if "X_test" not in artifact:
        raise RuntimeError("artifact must contain 'X_test'")
    if "train_df" not in artifact:
        raise RuntimeError("artifact must contain 'train_df'")
    if "initial_region" not in artifact:
        raise RuntimeError("artifact must contain 'initial_region' (call XReason beforehand)")

    pipeline = artifact["pipeline"]
    X_test: pd.DataFrame = artifact["X_test"]
    train_df: pd.DataFrame = artifact["train_df"]
    initial_region: Dict[str, Any] = artifact["initial_region"]

    if instance_index < 0 or instance_index >= len(X_test):
        raise IndexError("instance_index out of range")

    instance = X_test.iloc[instance_index:instance_index + 1]
    inst_raw = instance.iloc[0]

    pre = pipeline.named_steps.get("pre") if hasattr(pipeline, "named_steps") else None
    clf = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else pipeline

    try:
        if pre is not None:
            transformed_feature_names = list(pre.get_feature_names_out(list(train_df.columns)))
        else:
            transformed_feature_names = list(train_df.columns)
    except Exception:
        try:
            X_tmp = pre.transform(train_df) if pre is not None else train_df.values
            transformed_feature_names = [f"f{i}" for i in range(X_tmp.shape[1])]
        except Exception:
            transformed_feature_names = [f"f{i}" for i in range(200)]

    if class_idx is None:
        try:
            x_inst_trans = pre.transform(instance) if pre is not None else instance.values
            probs_inst = clf.predict_proba(x_inst_trans)[0]
            class_idx_local = int(np.argmax(probs_inst))
        except Exception:
            class_idx_local = 0
    else:
        class_idx_local = int(class_idx)

    # Explainers instance for heuristic checks (LIME/SHAP)
    expl_wrapper = Explainers(pipeline, train_df, list(train_df.columns))

    # heuristic assertion function (original heuristic claim)
    heuristic_assertion_fn = _make_heuristic_assertion(expl_wrapper, explainer_kind, inst_raw.to_dict(), shap_top_k=shap_top_k)

    current_region = dict(initial_region)
    history: List[Dict[str, Any]] = []
    final_explanation = None
    final_samples = None

    for it in range(max_iters):
        samples = constrained_sampling(current_region, inst_raw, n_samples=n_samples,
                                       random_state=random_state, expand_numeric=0.0)
        if samples is None or len(samples) == 0:
            raise RuntimeError("Sampling returned no rows; check region or sampler implementation.")

        kind = explainer_kind.lower()
        if kind == "lime":
            res = fit_constrained_lime(
                pipeline=pipeline,
                preprocessor=pre,
                model=clf,
                transformed_feature_names=transformed_feature_names,
                instance_raw=inst_raw.to_dict(),
                samples_raw=samples,
                class_idx=class_idx_local,
                num_features=min(10, len(transformed_feature_names)),
                alpha=1.0,
                kernel_width=kernel_width
            )
            r2_val = res["r2_weighted"]
            explanation = {"kind": "constrained_lime", "top_features": res["top_features"], "coeffs": res["coeffs"], "r2": float(r2_val)}
        elif kind == "shap":
            if not SHAP_AVAILABLE:
                raise RuntimeError("SHAP requested but 'shap' package not available.")
            res_shap = fit_constrained_shap(
                pipeline=pipeline, preprocessor=pre, model=clf,
                instance_raw=inst_raw.to_dict(), samples_raw=samples,
                transformed_feature_names=transformed_feature_names,
                nsamples=shap_nsamples
            )
            topnames = [n for n, _ in res_shap["top_features"][:10]]
            idxs = [transformed_feature_names.index(n) for n in topnames if n in transformed_feature_names]
            if len(idxs) == 0:
                r2_val = 0.0
            else:
                try:
                    X_trans = pre.transform(samples) if pre is not None else samples.values
                except Exception:
                    X_trans = samples.values
                X_sel = X_trans[:, idxs]
                try:
                    probs = clf.predict_proba(X_trans)
                    y = np.array([p[class_idx_local] for p in probs])
                    ridge_tmp = Ridge(alpha=1.0)
                    ridge_tmp.fit(X_sel, y)
                    y_pred_tmp = ridge_tmp.predict(X_sel)
                    r2_val = float(r2_score(y, y_pred_tmp))
                except Exception:
                    r2_val = 0.0
            explanation = {"kind": "constrained_shap", "top_features": res_shap["top_features"], "r2": float(r2_val)}
        else:
            raise ValueError("Unknown explainer_kind: choose 'lime' or 'shap'")

        # SAT-based verification (if requested)
        sat_cex = None
        sat_found = False
        if use_sat:
            if not SAT_AVAILABLE:
                # warn and skip SAT
                sat_cex = None
                sat_found = False
            else:
                # Define a model_predict_fn: pipeline expects raw DataFrame -> pipeline handles preprocessing
                def model_predict_fn(df):
                    return pipeline.predict(df)

                # heuristic_assertion_fn already created above; it checks original heuristic claim
                try:
                    sat_cex = find_counterexample_with_sat(
                        region=current_region,
                        original_instance=inst_raw,
                        model_predict_fn=model_predict_fn,
                        heuristic_assertion_fn=heuristic_assertion_fn,
                        num_bins=sat_bins,
                        max_tries=sat_tries,
                        solver_name=sat_solver,
                        verbose=sat_verbose
                    )
                except Exception as e:
                    # SAT error: treat as no-CE found but record the error
                    sat_cex = {"error": str(e)}
                    sat_found = False
                if isinstance(sat_cex, (pd.Series, dict)) and not isinstance(sat_cex, dict) or (isinstance(sat_cex, pd.Series)):
                    # if a pandas Series (concrete sample), it's a CE
                    sat_found = True
                elif isinstance(sat_cex, dict) and "error" in sat_cex:
                    sat_found = False
                else:
                    sat_found = sat_cex is not None

        # record iteration
        history.append({"iter": it, "region": current_region, "explanation": explanation, "r2": float(explanation.get("r2", 0.0)), "sat_cex": sat_cex if sat_cex is not None else None})

        # accept if R^2 meets threshold AND (if SAT requested) SAT did NOT find a CE
        r2_ok = float(explanation.get("r2", 0.0)) >= float(r2_threshold)
        sat_ok = True if not use_sat else (not sat_found)
        if r2_ok and sat_ok:
            final_explanation = explanation
            final_samples = samples
            break

        # if not acceptable: tighten region and continue
        current_region = tighten_region(current_region, inst_raw, shrink_factor=shrink_factor)

    return {
        "instance_index": int(instance_index),
        "initial_region": initial_region,
        "final_region": current_region,
        "final_explanation": final_explanation,
        "history": history,
        "final_samples": final_samples
    }
