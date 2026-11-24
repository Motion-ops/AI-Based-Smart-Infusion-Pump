"""
report_metrics_v2.py — Smart Infusion AI

Fixes the action-label lookup (it is `AI_Action`) and adds two checks that
matter for the progress seminar:

  1. Prints the exact feature list each model was trained on, so we can
     verify that SVR / CO / DrugEffect / PlasmaConcentration / Profile /
     SAPS_II were genuinely withheld (the partial-observability claim).

  2. Scores the flow regressor against BOTH `FlowRate` and `RecommendedFlow`
     and flags whichever one is also an input feature (that would be
     target leakage, and the score against it would be meaningless).

  3. Prints the clinical-profile mix of the train and test sets, to explain
     the difference in how often deterioration occurs between them.

HOW TO RUN
----------
Put this in your project root (the folder with `data/` and `models/`) and run:

    C:\\Python314\\python.exe report_metrics_v2.py

Paste everything it prints back into the chat.
Nothing is modified — this only reads.
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
TEST_CSV = os.path.join("data", "test_patients_large.csv")

MODELS = {
    "action": os.path.join("models", "ai_action_classifier.pkl"),
    "flow": os.path.join("models", "ai_flow_regressor.pkl"),
    "risk": os.path.join("models", "risk_classifier.pkl"),
}

ACTION_TARGET = "AI_Action"
FLOW_TARGETS = ["RecommendedFlow", "FlowRate"]
RISK_TARGET = "WillDeteriorate"
PROFILE_COL = "Profile"

# These should NOT appear in any model's feature list.
SHOULD_BE_HIDDEN = [
    "SVR", "CO", "DrugEffect", "PlasmaConcentration",
    "Profile", "SAPS_II", "ClearanceLMin", "VdL",
    "MAP", "HR", "SpO2",          # true (un-noised) values
    "ClinicalExplanation", "WillDeteriorate",
]


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


def feature_names(model):
    names = getattr(model, "feature_names_in_", None)
    return list(names) if names is not None else None


def align(model, df):
    names = feature_names(model)
    if names is None:
        print("  !! Model stores no feature names — cannot align safely.")
        return None
    missing = [n for n in names if n not in df.columns]
    if missing:
        print(f"  !! CSV missing model features: {missing}")
        return None
    return df[names]


def report_features(tag, model):
    names = feature_names(model)
    print(f"\n  Features used by {tag} ({len(names) if names else '?'}):")
    if not names:
        return
    for n in names:
        print(f"      {n}")
    leaked = [n for n in names if n in SHOULD_BE_HIDDEN]
    if leaked:
        print(f"\n  ** WARNING — these were supposed to be hidden: {leaked}")
    else:
        print("\n  OK — no hidden/unmeasurable variables present.")


def main():
    print("=" * 64)
    print("SMART INFUSION AI — TRAIN vs TEST (v2)")
    print("=" * 64)

    for p in (TRAIN_CSV, TEST_CSV):
        if not os.path.exists(p):
            sys.exit(f"Could not find {p}. Run from your project root.")

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"\nTrain rows: {len(train):,}    Test rows: {len(test):,}")

    # ---------- population mix ----------
    print("\n" + "-" * 64)
    print("POPULATION MIX  (explains differing deterioration rates)")
    print("-" * 64)
    mix = {}
    for name, df in (("train", train), ("test", test)):
        print(f"\n  {name.upper()}")
        counts = df[PROFILE_COL].value_counts()
        pats = df.groupby(PROFILE_COL)["PatientID"].nunique()
        for prof in counts.index:
            share = counts[prof] / len(df) * 100
            mix[f"{name}_{prof}_pct"] = round(float(share), 2)
            mix[f"{name}_{prof}_patients"] = int(pats.get(prof, 0))
            print(f"    {str(prof):<10} {share:5.1f}% of rows   {pats.get(prof,0):>4} patients")
        rate = df[RISK_TARGET].mean() * 100
        mix[f"{name}_deterioration_pct"] = round(float(rate), 2)
        print(f"    deterioration rate: {rate:.2f}%")

    results = {"population": mix}

    # ---------- action classifier ----------
    print("\n" + "-" * 64)
    print("ACTION CLASSIFIER")
    print("-" * 64)
    model = load_model(MODELS["action"])
    if model is not None:
        report_features("action classifier", model)
        block = {}
        for name, df in (("train", train), ("test", test)):
            X = align(model, df)
            if X is None:
                break
            pred = model.predict(X)
            acc = accuracy_score(df[ACTION_TARGET], pred)
            block[f"{name}_overall"] = round(float(acc) * 100, 2)
            print(f"\n  {name.upper()} overall accuracy: {acc*100:.2f}%")
            for prof in sorted(df[PROFILE_COL].dropna().unique()):
                m = (df[PROFILE_COL] == prof).values
                a = accuracy_score(df.loc[m, ACTION_TARGET], pred[m])
                block[f"{name}_{prof}"] = round(float(a) * 100, 2)
                print(f"    {str(prof):<10} {a*100:6.2f}%   (n={int(m.sum()):,})")
        # class balance of the label itself
        print("\n  Action label distribution (train):")
        for k, v in train[ACTION_TARGET].value_counts(normalize=True).items():
            block[f"label_{k}_pct"] = round(float(v) * 100, 2)
            print(f"    {str(k):<12} {v*100:5.2f}%")
        results["action"] = block

    # ---------- flow regressor ----------
    print("\n" + "-" * 64)
    print("FLOW REGRESSOR")
    print("-" * 64)
    model = load_model(MODELS["flow"])
    if model is not None:
        report_features("flow regressor", model)
        feats = feature_names(model) or []
        block = {}
        for target in FLOW_TARGETS:
            if target not in train.columns:
                continue
            leak = target in feats
            print(f"\n  Target = {target}" + ("   ** ALSO AN INPUT FEATURE — INVALID **" if leak else ""))
            block[f"{target}_is_feature"] = bool(leak)
            for name, df in (("train", train), ("test", test)):
                X = align(model, df)
                if X is None:
                    break
                pred = model.predict(X)
                mae = mean_absolute_error(df[target], pred)
                r2 = r2_score(df[target], pred)
                block[f"{target}_{name}_mae"] = round(float(mae), 4)
                block[f"{target}_{name}_r2"] = round(float(r2), 5)
                print(f"    {name.upper():<6} MAE {mae:8.4f} mL/hr    R2 {r2:.5f}")
        results["flow"] = block

    # ---------- risk model ----------
    print("\n" + "-" * 64)
    print("RISK MODEL")
    print("-" * 64)
    model = load_model(MODELS["risk"])
    if model is not None:
        report_features("risk model", model)
        block = {}
        for name, df in (("train", train), ("test", test)):
            X = align(model, df)
            if X is None:
                break
            y = df[RISK_TARGET].astype(int)
            pred = model.predict(X)
            block[f"{name}_precision"] = round(float(precision_score(y, pred, zero_division=0)), 4)
            block[f"{name}_recall"] = round(float(recall_score(y, pred, zero_division=0)), 4)
            block[f"{name}_f1"] = round(float(f1_score(y, pred, zero_division=0)), 4)
            try:
                block[f"{name}_roc_auc"] = round(float(roc_auc_score(y, model.predict_proba(X)[:, 1])), 5)
            except Exception:
                pass
            block[f"{name}_positive_rate"] = round(float(y.mean()) * 100, 2)
            print(f"\n  {name.upper()}")
            for k, v in block.items():
                if k.startswith(name):
                    print(f"    {k.replace(name + '_', ''):<14} {v}")
        results["risk"] = block

    print("\n" + "=" * 64)
    print("COPY EVERYTHING BELOW INTO THE CHAT")
    print("=" * 64)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
