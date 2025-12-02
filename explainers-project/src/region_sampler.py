# src/region_sampler.py
import numpy as np
import pandas as pd
from typing import Dict, Any

def sample_from_region(region_constraints: Dict[str, Any], original_instance: pd.Series, n_samples: int = 500, random_state: int = 42, expand_numeric: float = 0.0) -> pd.DataFrame:
    """
    region_constraints: dict mapping raw_feature -> ["eq", value] or [low, high]
      - If ["eq", None] -> use original_instance value (fix)
      - If ["eq", value] -> set that value
      - If [low, high] numeric -> sample uniformly between low and high
    expand_numeric: relative expansion of numeric intervals (0.0 means no expansion).
    """
    rng = np.random.default_rng(random_state)
    feat_names = original_instance.index.tolist()
    rows = []
    for i in range(n_samples):
        s = original_instance.copy()
        for f, cons in region_constraints.items():
            if f not in s.index:
                continue
            # equality with explicit value
            if isinstance(cons, list) and len(cons) == 2 and cons[0] == "eq":
                val = cons[1]
                if val is None:
                    # preserve original value
                    s[f] = original_instance[f]
                else:
                    s[f] = val
            # numeric interval [low, high]
            elif isinstance(cons, list) and len(cons) == 2 and all(isinstance(x, (int, float)) for x in cons):
                lo, hi = float(cons[0]), float(cons[1])
                # expand interval if requested
                if expand_numeric != 0.0 and lo < hi:
                    width = hi - lo
                    lo = lo - expand_numeric * width
                    hi = hi + expand_numeric * width
                if abs(hi - lo) < 1e-9:
                    s[f] = lo
                else:
                    # choose float or int depending on original type
                    orig = original_instance[f]
                    if pd.api.types.is_integer_dtype(type(orig)) or isinstance(orig, int):
                        # integer sample
                        lowi = int(np.floor(lo))
                        highi = int(np.ceil(hi))
                        if highi <= lowi:
                            s[f] = lowi
                        else:
                            s[f] = int(rng.integers(lowi, highi + 1))
                    else:
                        s[f] = float(rng.uniform(lo, hi))
            else:
                # unknown constraint format -> keep original
                s[f] = original_instance[f]
        rows.append(s.values)
    df = pd.DataFrame(rows, columns=feat_names)
    return df

def test_counterexamples(samples_df: pd.DataFrame, model_predict_fn, heuristic_assertion_fn):
    """
    Iterate sample rows and return DataFrame of samples where heuristic_assertion_fn is False
    (i.e. counterexamples), but model prediction remains the same (model_predict_fn consistent).
    model_predict_fn(df) -> predicted labels (array-like) for the provided DataFrame.
    heuristic_assertion_fn(df_row_df) -> True if heuristic holds for this single-row DataFrame
    """
    counter_rows = []
    for i in range(len(samples_df)):
        row_df = samples_df.iloc[[i]]
        try:
            # allow model_predict_fn to accept raw df or transformed form
            pred = model_predict_fn(row_df)
        except Exception:
            # skip problematic rows
            continue
        try:
            holds = heuristic_assertion_fn(row_df)
        except Exception:
            holds = False
        if not holds:
            counter_rows.append(row_df.iloc[0])
    if counter_rows:
        return pd.DataFrame(counter_rows).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=samples_df.columns)
# src/region_sampler.py
import numpy as np
import pandas as pd
from typing import Dict, Any

def sample_from_region(region_constraints: Dict[str, Any], original_instance: pd.Series, n_samples: int = 500, random_state: int = 42, expand_numeric: float = 0.0) -> pd.DataFrame:
    """
    region_constraints: dict mapping raw_feature -> ["eq", value] or [low, high]
      - If ["eq", None] -> use original_instance value (fix)
      - If ["eq", value] -> set that value
      - If [low, high] numeric -> sample uniformly between low and high
    expand_numeric: relative expansion of numeric intervals (0.0 means no expansion).
    """
    rng = np.random.default_rng(random_state)
    feat_names = original_instance.index.tolist()
    rows = []
    for i in range(n_samples):
        s = original_instance.copy()
        for f, cons in region_constraints.items():
            if f not in s.index:
                continue
            # equality with explicit value
            if isinstance(cons, list) and len(cons) == 2 and cons[0] == "eq":
                val = cons[1]
                if val is None:
                    # preserve original value
                    s[f] = original_instance[f]
                else:
                    s[f] = val
            # numeric interval [low, high]
            elif isinstance(cons, list) and len(cons) == 2 and all(isinstance(x, (int, float)) for x in cons):
                lo, hi = float(cons[0]), float(cons[1])
                # expand interval if requested
                if expand_numeric != 0.0 and lo < hi:
                    width = hi - lo
                    lo = lo - expand_numeric * width
                    hi = hi + expand_numeric * width
                if abs(hi - lo) < 1e-9:
                    s[f] = lo
                else:
                    # choose float or int depending on original type
                    orig = original_instance[f]
                    if pd.api.types.is_integer_dtype(type(orig)) or isinstance(orig, int):
                        # integer sample
                        lowi = int(np.floor(lo))
                        highi = int(np.ceil(hi))
                        if highi <= lowi:
                            s[f] = lowi
                        else:
                            s[f] = int(rng.integers(lowi, highi + 1))
                    else:
                        s[f] = float(rng.uniform(lo, hi))
            else:
                # unknown constraint format -> keep original
                s[f] = original_instance[f]
        rows.append(s.values)
    df = pd.DataFrame(rows, columns=feat_names)
    return df

def test_counterexamples(samples_df: pd.DataFrame, model_predict_fn, heuristic_assertion_fn):
    """
    Iterate sample rows and return DataFrame of samples where heuristic_assertion_fn is False
    (i.e. counterexamples), but model prediction remains the same (model_predict_fn consistent).
    model_predict_fn(df) -> predicted labels (array-like) for the provided DataFrame.
    heuristic_assertion_fn(df_row_df) -> True if heuristic holds for this single-row DataFrame
    """
    counter_rows = []
    for i in range(len(samples_df)):
        row_df = samples_df.iloc[[i]]
        try:
            # allow model_predict_fn to accept raw df or transformed form
            pred = model_predict_fn(row_df)
        except Exception:
            # skip problematic rows
            continue
        try:
            holds = heuristic_assertion_fn(row_df)
        except Exception:
            holds = False
        if not holds:
            counter_rows.append(row_df.iloc[0])
    if counter_rows:
        return pd.DataFrame(counter_rows).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=samples_df.columns)
