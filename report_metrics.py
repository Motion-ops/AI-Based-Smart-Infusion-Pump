"""
report_metrics.py — Smart Infusion AI
Scores the trained models on BOTH the training and test sets so the
progress-seminar slides can show a train-vs-test comparison.

HOW TO RUN
----------
Put this file in your project root (the folder containing `data/` and `models/`)
and run:

    C:\\Python314\\python.exe report_metrics.py

Then copy EVERYTHING it prints and paste it back into the chat.

Nothing is modified. This only reads your CSVs and .pkl files.
"""

import os
import sys
import json

try:
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import (
        accuracy_score, mean_absolute_error, r2_score,
        precision_score, recall_score, f1_score, roc_auc_score,
    )
except ImportError as e:
    sys.exit(f"Missing package: {e}. Install with: pip install pandas scikit-learn joblib")

TRAIN_CSV = os.path.join("data", "train_patients.csv")
TEST_CSV = os.path.join("data", "test_patients.csv")

MODELS = {
    "action": os.path.join("models", "ai_action_classifier.pkl"),
    "flow": os.path.join("models", "ai_flow_regressor.pkl"),
    "risk": os.path.join("models", "risk_classifier.pkl"),
}

# The script tries these column names in order until one is found.
TARGET_CANDIDATES = {
    "action": ["Action", "ControllerAction", "ActionLabel", "Mode", "ControllerMode"],
    "flow": ["Flow", "FlowRate", "ControllerFlow", "Flow_mL_hr", "FlowRate_mL_hr"],
    "risk": ["WillDeteriorate", "Deteriorate", "RiskLabel"],
}
PROFILE_CANDIDATES = ["Profile", "ClinicalProfile", "Severity"]


def find_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    print(f"  !! Could not find a '{what}' column. Tried: {candidates}")
    print(f"     Available columns: {list(df.columns)}")
    return None


def load_model(path):
    if not os.path.exists(path):
        print(f"  !! Model not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as fh:
            return pickle.load(fh)


def features_for(model, df):
    """Use the feature list the model was trained on."""
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        print("  !! Model has no stored feature names — cannot align columns safely.")
        return None
    missing = [n for n in names if n not in df.columns]
    if missing:
        print(f"  !! CSV is missing model features: {missing}")
        return None
    return df[list(names)]


def main():
    print("=" * 62)
    print("SMART INFUSION AI — TRAIN vs TEST METRICS")
    print("=" * 62)

    for p in (TRAIN_CSV, TEST_CSV):
        if not os.path.exists(p):
            sys.exit(f"Could not find {p}. Run this from your project root folder.")

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"\nTrain rows: {len(train):,}   Test rows: {len(test):,}")

    profile_col = find_col(train, PROFILE_CANDIDATES, "profile")
    results = {}

    # ---------------- ACTION CLASSIFIER ----------------
    print("\n" + "-" * 62)
    print("ACTION CLASSIFIER (imitates the controller's decision)")
    print("-" * 62)
    model = load_model(MODELS["action"])
    if model is not None:
        col = find_col(train, TARGET_CANDIDATES["action"], "action target")
        if col:
            block = {}
            for name, df in (("train", train), ("test", test)):
                X = features_for(model, df)
                if X is None:
                    break
                pred = model.predict(X)
                acc = accuracy_score(df[col], pred)
                block[f"{name}_overall"] = round(float(acc) * 100, 2)
                print(f"\n  {name.upper()} overall accuracy: {acc*100:.2f}%")
                if profile_col:
                    for prof in sorted(df[profile_col].dropna().unique()):
                        m = df[profile_col] == prof
                        a = accuracy_score(df.loc[m, col], pred[m.values])
                        block[f"{name}_{prof}"] = round(float(a) * 100, 2)
                        print(f"    {str(prof):<10} {a*100:6.2f}%   (n={int(m.sum()):,})")
            results["action"] = block

    # ---------------- FLOW REGRESSOR ----------------
    print("\n" + "-" * 62)
    print("FLOW REGRESSOR (predicts the controller's mL/hr)")
    print("-" * 62)
    model = load_model(MODELS["flow"])
    if model is not None:
        col = find_col(train, TARGET_CANDIDATES["flow"], "flow target")
        if col:
            block = {}
            for name, df in (("train", train), ("test", test)):
                X = features_for(model, df)
                if X is None:
                    break
                pred = model.predict(X)
                mae = mean_absolute_error(df[col], pred)
                r2 = r2_score(df[col], pred)
                block[f"{name}_mae"] = round(float(mae), 4)
                block[f"{name}_r2"] = round(float(r2), 5)
                print(f"  {name.upper():<6} MAE: {mae:.4f} mL/hr    R2: {r2:.5f}")
            results["flow"] = block

    # ---------------- RISK MODEL ----------------
    print("\n" + "-" * 62)
    print("RISK MODEL (forecasts deterioration — genuine prediction)")
    print("-" * 62)
    model = load_model(MODELS["risk"])
    if model is not None:
        col = find_col(train, TARGET_CANDIDATES["risk"], "risk target")
        if col:
            block = {}
            for name, df in (("train", train), ("test", test)):
                X = features_for(model, df)
                if X is None:
                    break
                y = df[col].astype(int)
                pred = model.predict(X)
                block[f"{name}_accuracy"] = round(float(accuracy_score(y, pred)) * 100, 2)
                block[f"{name}_precision"] = round(float(precision_score(y, pred, zero_division=0)), 4)
                block[f"{name}_recall"] = round(float(recall_score(y, pred, zero_division=0)), 4)
                block[f"{name}_f1"] = round(float(f1_score(y, pred, zero_division=0)), 4)
                try:
                    proba = model.predict_proba(X)[:, 1]
                    block[f"{name}_roc_auc"] = round(float(roc_auc_score(y, proba)), 5)
                except Exception:
                    pass
                print(f"\n  {name.upper()}")
                for k, v in block.items():
                    if k.startswith(name):
                        print(f"    {k.replace(name + '_', ''):<12} {v}")
                pos = float(y.mean()) * 100
                block[f"{name}_positive_rate"] = round(pos, 2)
                print(f"    {'positive %':<12} {pos:.2f}   (how often deterioration actually happens)")
            results["risk"] = block

    print("\n" + "=" * 62)
    print("COPY THE BLOCK BELOW AND PASTE IT INTO THE CHAT")
    print("=" * 62)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
