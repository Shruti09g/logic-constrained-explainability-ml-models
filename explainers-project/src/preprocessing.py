# src/preprocessing.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, List

def build_preprocessor(df: pd.DataFrame, categorical_thresh: int = 50) -> Tuple[ColumnTransformer, List[str]]:
    """
    Build a ColumnTransformer:
      - OneHotEncode object/string columns or low-cardinality columns
      - pass through numeric columns (remainder='passthrough')
    Returns (preprocessor, raw_feature_names)
    """
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].nunique() <= categorical_thresh]
    # numeric columns are the rest
    num_cols = [c for c in df.columns if c not in cat_cols]

    transformers = []
    if cat_cols:
        # sklearn >=1.4 uses sparse_output param
        transformers.append(("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

    from sklearn.compose import ColumnTransformer
    pre = ColumnTransformer(transformers, remainder="passthrough")
    return pre, list(df.columns)
