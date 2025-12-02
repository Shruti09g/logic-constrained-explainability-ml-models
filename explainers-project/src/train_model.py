# src/train_model.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from preprocessing import build_preprocessor
from utils import ensure_dir

DEFAULT_MODEL_PATH = "/home/vivresavie/explainers-project/models/rf_adult.joblib"

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def build_and_train(df: pd.DataFrame, target_col: str = "income", model_path: str = DEFAULT_MODEL_PATH, test_size=0.2, random_state=42):
    ensure_dir(os.path.dirname(model_path) or ".")
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # convert common binary target strings to 0/1 if needed
    if y_raw.dtype == object:
        y = (y_raw.astype(str).str.contains(">50") | y_raw.astype(str).str.contains("1")).astype(int)
    else:
        y = y_raw.astype(int)

    pre, feature_names = build_preprocessor(X)
    clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=random_state, n_jobs=-1)
    pipeline = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Trained RF: test accuracy = {acc:.4f}")

    artifact = {
        "pipeline": pipeline,
        "train_df": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True).tolist(),
        "y_test": y_test.reset_index(drop=True).tolist(),
        "feature_names": list(X.columns)
    }
    joblib.dump(artifact, model_path)
    print(f"Saved model artifact to {model_path}")
    return model_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/vivresavie/explainers-project/data/raw/adult.csv", help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="income", help="Target column name")
    parser.add_argument("--out", type=str, default=DEFAULT_MODEL_PATH, help="Model artifact path")
    args = parser.parse_args()

    df = load_csv(args.data)
    build_and_train(df, target_col=args.target, model_path=args.out)
