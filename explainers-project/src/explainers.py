# src/explainers.py
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import shap

class Explainers:
    """
    Wrapper providing LIME and SHAP explanations over a pipeline.

    Initialize with:
        pipeline: sklearn Pipeline (expects named_steps 'pre' and 'clf')
        train_df: pandas DataFrame (raw training data, before preprocessing)
        raw_feature_names: list of raw column names
    """
    def __init__(self, pipeline, train_df: pd.DataFrame, raw_feature_names):
        self.pipeline = pipeline
        # extract preprocessor and classifier (if pipeline has named steps)
        if hasattr(pipeline, "named_steps") and "pre" in pipeline.named_steps and "clf" in pipeline.named_steps:
            self.preprocessor = pipeline.named_steps["pre"]
            self.model = pipeline.named_steps["clf"]
        else:
            # fallback
            self.preprocessor = None
            self.model = pipeline

        self.train_df = train_df.copy().reset_index(drop=True)
        self.feature_names = list(raw_feature_names)

        # precompute transformed training matrix for LIME/SHAP background
        try:
            if self.preprocessor is not None:
                self.X_train_trans = self.preprocessor.transform(self.train_df)
            else:
                self.X_train_trans = self.train_df.values
        except Exception:
            self.X_train_trans = self.train_df.values

        # try to obtain transformed feature names
        try:
            if self.preprocessor is not None:
                self.transformed_feature_names = list(self.preprocessor.get_feature_names_out(self.feature_names))
            else:
                self.transformed_feature_names = list(self.feature_names)
        except Exception:
            self.transformed_feature_names = [f"f{i}" for i in range(self.X_train_trans.shape[1])]

    # ---------------- LIME ----------------
    def lime_explain(self, instance_raw, num_features: int = 5, random_state: int = None):
        """
        Run LIME and return the Explanation object (supports .as_list()).
        instance_raw: dict, pandas Series, or one-row DataFrame with raw columns.
        """
        # build instance dataframe in raw column order
        if isinstance(instance_raw, pd.Series):
            instance_df = pd.DataFrame([instance_raw.values], columns=instance_raw.index)
        elif isinstance(instance_raw, dict):
            instance_df = pd.DataFrame([instance_raw], columns=self.feature_names)
        elif isinstance(instance_raw, pd.DataFrame):
            instance_df = instance_raw
        else:
            instance_df = pd.DataFrame([instance_raw])

        # transform
        try:
            inst_trans = self.preprocessor.transform(instance_df) if self.preprocessor is not None else instance_df.values
        except Exception:
            inst_trans = instance_df.values

        # heuristic: detect categorical transformed features (one-hot 0/1)
        categorical_features = []
        try:
            for idx in range(self.X_train_trans.shape[1]):
                col = self.X_train_trans[:, idx]
                unique_vals = np.unique(col)
                if np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [1]) or np.array_equal(unique_vals, [0, 1]):
                    categorical_features.append(idx)
        except Exception:
            categorical_features = []

        explainer = LimeTabularExplainer(
            training_data=self.X_train_trans,
            feature_names=self.transformed_feature_names,
            categorical_features=categorical_features,
            mode="classification",
            random_state=random_state
        )

        # LIME will send transformed arrays; classifier expects transformed arrays
        def predict_fn(x):
            return self.model.predict_proba(x)

        explanation = explainer.explain_instance(inst_trans[0], predict_fn, num_features=num_features)
        return explanation

    # ---------------- SHAP ----------------
    def shap_explain(self, instance_raw, top_k: int = 6):
        """
        Return top_k list of (transformed_feature_name, shap_value) pairs.
        Uses TreeExplainer when possible; flattens arrays to floats.
        """
        # prepare instance as DataFrame
        if isinstance(instance_raw, pd.Series):
            instance_df = pd.DataFrame([instance_raw.values], columns=instance_raw.index)
        elif isinstance(instance_raw, dict):
            instance_df = pd.DataFrame([instance_raw], columns=self.feature_names)
        elif isinstance(instance_raw, pd.DataFrame):
            instance_df = instance_raw
        else:
            instance_df = pd.DataFrame([instance_raw])

        try:
            inst_trans = self.preprocessor.transform(instance_df) if self.preprocessor is not None else instance_df.values
        except Exception:
            inst_trans = instance_df.values

        # Try TreeExplainer (fast for tree ensembles); otherwise fallback to KernelExplainer
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_vals = explainer.shap_values(inst_trans)
        except Exception:
            # fallback: kernel explainer with small background
            try:
                bg = self.X_train_trans if self.X_train_trans.shape[0] <= 100 else self.X_train_trans[np.random.choice(self.X_train_trans.shape[0], 100, replace=False)]
                explainer = shap.KernelExplainer(lambda x: self.model.predict_proba(x), bg)
                shap_vals = explainer.shap_values(inst_trans, nsamples=100)
            except Exception as e:
                raise RuntimeError(f"SHAP explainer error: {e}")

        # select class index with highest predicted probability
        try:
            probs = self.model.predict_proba(inst_trans)[0]
            class_idx = int(np.argmax(probs))
        except Exception:
            class_idx = 0

        if isinstance(shap_vals, list):
            vals = shap_vals[class_idx]
        else:
            vals = shap_vals

        # ensure flatten and convert to floats
        vals_row = np.array(vals).reshape(-1)  # flatten
        # zip with transformed feature names (if lengths mismatch, truncate/pad)
        n = min(len(self.transformed_feature_names), vals_row.shape[0])
        feats = [(self.transformed_feature_names[i], float(vals_row[i])) for i in range(n)]
        feats_sorted = sorted(feats, key=lambda x: -abs(x[1]))
        return feats_sorted[:top_k]
