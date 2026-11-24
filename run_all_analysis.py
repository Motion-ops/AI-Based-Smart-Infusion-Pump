"""
===============================================================================
RUN_ALL_ANALYSIS.PY
Smart AI Infusion Pump - complete evaluation in one run

Combines everything into a single script:
  1. Population mix of train vs test
  2. Action classifier: overall + per-profile, train vs test
  3. Flow regressor: MAE / R2, train vs test
  4. Risk model: precision / recall / F1 / ROC-AUC, train vs test
  5. Feature-list check (verifies hidden variables really are hidden)
  6. Confusion matrix + clinical error-direction analysis
  7. Weaning diagnostics on the long-episode set (if present)

Nothing is retrained. Nothing is modified. Read-only.

HOW TO RUN
----------
From your project root:

    C:\\Python314\\python.exe run_all_analysis.py

Any dataset that does not exist is skipped with a note, so this runs
even if you have not generated the long set yet.

Paste the whole output back into the chat.
===============================================================================
"""

import json
import os
import sys

try:
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.metrics import (
        accuracy_score, mean_absolute_error, r2_score,
        precision_score, recall_score, f1_score, roc_auc_score,
        confusion_matrix, classification_report,
    )
except ImportError as e:
    sys.exit(f"Missing package: {e}. Install with: pip install pandas scikit-learn joblib")

# ----------------------------------------------------------------------
TRAIN_CSV = os.path.join("data", "train_patients.csv")
TEST_CSV = os.path.join("data", "test_patients_large.csv")   # 300-patient set
FALLBACK_TEST = os.path.join("data", "test_patients.csv")    # old 60-patient set
LONG_CSV = os.path.join("data", "test_patients_long.csv")

MODELS = {
    "action": os.path.join("models", "ai_action_classifier.pkl"),
    "flow": os.path.join("models", "ai_flow_regressor.pkl"),
    "risk": os.path.join("models", "risk_classifier.pkl"),
}

ACTION_LABEL = "AI_Action"
FLOW_LABEL = "RecommendedFlow"     # the controller's DECISION (not FlowRate)
RISK_LABEL = "WillDeteriorate"
PROFILE_COL = "Profile"

ACTION_ORDER = ["Decrease", "Maintain", "Increase", "Emergency"]

SHOULD_BE_HIDDEN = [
    "SVR", "CO", "DrugEffect", "PlasmaConcentration",
    "Profile", "SAPS_II", "ClearanceLMin", "VdL",
    "MAP", "HR", "SpO2", "ClinicalExplanation", "WillDeteriorate",
]

RESULTS = {}


# ----------------------------------------------------------------------
def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def sub(title):
    print("\n" + "-" * 72)
    print(title)
    print("-" * 72)


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


def feats(model):
    n = getattr(model, "feature_names_in_", None)
    return list(n) if n is not None else None


def align(model, df):
    names = feats(model)
    if names is None:
        print("  !! Model stores no feature names.")
        return None
    missing = [n for n in names if n not in df.columns]
    if missing:
        print(f"  !! CSV missing features: {missing}")
        return None
    return df[names]


# ----------------------------------------------------------------------
def section_population(train, test):
    hr("1. POPULATION MIX")
    out = {}
    for name, df in (("train", train), ("test", test)):
        print(f"\n  {name.upper()}  ({len(df):,} rows)")
        pats = df.groupby(PROFILE_COL)["PatientID"].nunique()
        for prof in ["Healthy", "Mild", "Moderate", "Severe"]:
            if prof in pats.index:
                n = int(pats[prof])
                out[f"{name}_{prof}_patients"] = n
                print(f"    {prof:<10} {n:>5} patients")
        rate = float(df[RISK_LABEL].mean()) * 100
        out[f"{name}_deterioration_pct"] = round(rate, 2)
        print(f"    deterioration rate: {rate:.2f}%")
    RESULTS["population"] = out


def section_features(model, tag):
    names = feats(model)
    if not names:
        return
    leaked = [n for n in names if n in SHOULD_BE_HIDDEN]
    print(f"\n  {tag}: {len(names)} features")
    if leaked:
        print(f"  ** WARNING - should have been hidden: {leaked}")
    else:
        print("  OK - no hidden/unmeasurable variables used as inputs.")


def section_action(train, test):
    hr("2. ACTION CLASSIFIER")
    model = load_model(MODELS["action"])
    if model is None:
        return None, None, None
    section_features(model, "action classifier")

    out = {}
    preds = {}
    for name, df in (("train", train), ("test", test)):
        X = align(model, df)
        if X is None:
            return None, None, None
        pred = pd.Series(model.predict(X)).astype(str)
        preds[name] = pred
        acc = accuracy_score(df[ACTION_LABEL].astype(str), pred)
        out[f"{name}_overall"] = round(float(acc) * 100, 2)
        print(f"\n  {name.upper()} overall: {acc*100:.2f}%")
        for prof in ["Healthy", "Mild", "Moderate", "Severe"]:
            m = (df[PROFILE_COL] == prof).values
            if not m.any():
                continue
            a = accuracy_score(df.loc[m, ACTION_LABEL].astype(str), pred[m])
            out[f"{name}_{prof}"] = round(float(a) * 100, 2)
            print(f"    {prof:<10} {a*100:6.2f}%   (n={int(m.sum()):,})")

    print("\n  Gap (train - test) by profile:")
    for prof in ["Healthy", "Mild", "Moderate", "Severe"]:
        k1, k2 = f"train_{prof}", f"test_{prof}"
        if k1 in out and k2 in out:
            g = out[k1] - out[k2]
            out[f"gap_{prof}"] = round(g, 2)
            print(f"    {prof:<10} {g:+6.2f} points")

    RESULTS["action"] = out
    return model, preds["test"], test[ACTION_LABEL].astype(str)


def section_flow(train, test):
    hr("3. FLOW REGRESSOR")
    model = load_model(MODELS["flow"])
    if model is None:
        return
    section_features(model, "flow regressor")
    if FLOW_LABEL in (feats(model) or []):
        print(f"  ** WARNING: {FLOW_LABEL} is also an input feature - scores invalid.")
    out = {}
    for name, df in (("train", train), ("test", test)):
        X = align(model, df)
        if X is None:
            return
        pred = model.predict(X)
        mae = mean_absolute_error(df[FLOW_LABEL], pred)
        r2 = r2_score(df[FLOW_LABEL], pred)
        out[f"{name}_mae"] = round(float(mae), 4)
        out[f"{name}_r2"] = round(float(r2), 5)
        print(f"  {name.upper():<6} MAE {mae:7.4f} mL/hr    R2 {r2:.5f}")
    RESULTS["flow"] = out


def section_risk(train, test):
    hr("4. RISK MODEL")
    model = load_model(MODELS["risk"])
    if model is None:
        return
    section_features(model, "risk model")
    out = {}
    for name, df in (("train", train), ("test", test)):
        X = align(model, df)
        if X is None:
            return
        y = df[RISK_LABEL].astype(int)
        pred = model.predict(X)
        out[f"{name}_precision"] = round(float(precision_score(y, pred, zero_division=0)), 4)
        out[f"{name}_recall"] = round(float(recall_score(y, pred, zero_division=0)), 4)
        out[f"{name}_f1"] = round(float(f1_score(y, pred, zero_division=0)), 4)
        try:
            out[f"{name}_roc_auc"] = round(
                float(roc_auc_score(y, model.predict_proba(X)[:, 1])), 5)
        except Exception:
            pass
        out[f"{name}_positive_rate"] = round(float(y.mean()) * 100, 2)
        print(f"\n  {name.upper()}")
        for k, v in out.items():
            if k.startswith(name):
                print(f"    {k.replace(name+'_',''):<14} {v}")
    RESULTS["risk"] = out


def print_matrix(y_true, y_pred, labels, title):
    sub(title)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    w = max(len(l) for l in labels) + 2
    print("\n  Counts  (rows = controller's choice, cols = AI prediction)\n")
    print(" " * (w + 2) + "".join(f"{l:>12}" for l in labels) + f"{'total':>12}")
    for i, l in enumerate(labels):
        print(f"  {l:<{w}}" + "".join(f"{cm[i,j]:>12,}" for j in range(len(labels)))
              + f"{cm[i].sum():>12,}")
    print("\n  Row percentages\n")
    print(" " * (w + 2) + "".join(f"{l:>12}" for l in labels))
    for i, l in enumerate(labels):
        t = cm[i].sum()
        if t == 0:
            continue
        print(f"  {l:<{w}}" + "".join(f"{cm[i,j]/t*100:>11.1f}%" for j in range(len(labels))))
    rep = classification_report(y_true, y_pred, labels=labels,
                                zero_division=0, output_dict=True)
    print(f"\n  {'class':<12}{'precision':>11}{'recall':>10}{'f1':>10}{'support':>11}")
    for l in labels:
        if l in rep:
            r = rep[l]
            print(f"  {l:<12}{r['precision']:>11.3f}{r['recall']:>10.3f}"
                  f"{r['f1-score']:>10.3f}{int(r['support']):>11,}")
    return rep


def direction(y_true, y_pred, tag):
    rank = {l: i for i, l in enumerate(ACTION_ORDER)}
    t = pd.Series(list(y_true)).reset_index(drop=True)
    p = pd.Series(list(y_pred)).reset_index(drop=True)
    ok = t.isin(rank) & p.isin(rank)
    tr, pr = t[ok].map(rank), p[ok].map(rank)
    total, wrong = int(ok.sum()), int((tr != pr).sum())
    under, over = int((pr < tr).sum()), int((pr > tr).sum())

    sub(f"ERROR DIRECTION - {tag}")
    print(f"\n  Rows            : {total:,}")
    print(f"  Errors          : {wrong:,} ({wrong/max(total,1)*100:.2f}%)")
    print(f"  UNDER-dosing    : {under:,} ({under/max(wrong,1)*100:.1f}% of errors)")
    print(f"  Over-dosing     : {over:,} ({over/max(wrong,1)*100:.1f}% of errors)")

    mi = int(((t == "Increase") & p.isin(["Maintain", "Decrease"])).sum())
    ni = int((t == "Increase").sum())
    me = int(((t == "Emergency") & (p != "Emergency")).sum())
    ne = int((t == "Emergency").sum())
    print(f"\n  Controller said INCREASE,  AI said hold/reduce : "
          f"{mi:,}/{ni:,} ({mi/max(ni,1)*100:.1f}%)")
    print(f"  Controller said EMERGENCY, AI did not          : "
          f"{me:,}/{ne:,} ({me/max(ne,1)*100:.1f}%)")

    return {"rows": total, "errors": wrong, "under_dosing": under, "over_dosing": over,
            "under_pct_of_errors": round(under / max(wrong, 1) * 100, 2),
            "missed_increase": mi, "n_increase": ni,
            "missed_increase_pct": round(mi / max(ni, 1) * 100, 2),
            "missed_emergency": me, "n_emergency": ne,
            "missed_emergency_pct": round(me / max(ne, 1) * 100, 2)}


def section_confusion(test, y_pred, y_true):
    hr("5. CONFUSION + ERROR DIRECTION")
    labels = [l for l in ACTION_ORDER if l in set(y_true) | set(y_pred)]
    rep = print_matrix(y_true, y_pred, labels, "ALL TEST PATIENTS")
    RESULTS["per_class"] = {
        l: {"precision": round(rep[l]["precision"], 4),
            "recall": round(rep[l]["recall"], 4),
            "f1": round(rep[l]["f1-score"], 4),
            "support": int(rep[l]["support"])}
        for l in labels if l in rep}
    RESULTS["direction_all"] = direction(y_true, y_pred, "ALL TEST PATIENTS")

    m = (test[PROFILE_COL] == "Severe").values
    if m.any():
        ys = y_true[m].reset_index(drop=True)
        ps = y_pred[m].reset_index(drop=True)
        sev_labels = [l for l in labels if l in set(ys) | set(ps)]
        print("\n")
        hr("SEVERE PATIENTS ONLY")
        print(f"Accuracy: {accuracy_score(ys, ps)*100:.2f}%")
        rep_s = print_matrix(ys, ps, sev_labels, "SEVERE ONLY")
        RESULTS["severe_per_class"] = {
            l: {"precision": round(rep_s[l]["precision"], 4),
                "recall": round(rep_s[l]["recall"], 4),
                "f1": round(rep_s[l]["f1-score"], 4),
                "support": int(rep_s[l]["support"])}
            for l in sev_labels if l in rep_s}
        RESULTS["direction_severe"] = direction(ys, ps, "SEVERE ONLY")


def section_weaning():
    hr("6. WEANING (long-episode set)")
    if not os.path.exists(LONG_CSV):
        print(f"\n  {LONG_CSV} not found - skipped.")
        print("  Generate it with generate_eval_datasets.py --skip-large")
        return
    df = pd.read_csv(LONG_CSV)
    tmax = df["Time"].max()
    print(f"\n  {len(df):,} rows, {df['PatientID'].nunique()} patients, "
          f"{tmax+1} ticks ({(tmax+1)/60:.0f} simulated minutes)")

    early = df[df["Time"] < 300].groupby("PatientID")["FlowRate"].mean()
    late = df[df["Time"] > tmax - 300].groupby("PatientID")["FlowRate"].mean()
    j = pd.concat([early.rename("early"), late.rename("late")], axis=1).dropna()
    dosed = j[j["early"] > 0.1]

    weaned = int((dosed["late"] < dosed["early"] * 0.25).sum())
    zeroed = int((dosed["late"] <= 0.01).sum())
    print(f"\n  Patients that received drug early : {len(dosed)}")
    print(f"  Flow fell below 25% of early rate : {weaned}/{len(dosed)}")
    print(f"  Fully weaned to zero              : {zeroed}/{len(dosed)}")

    hr_e = df[df["Time"] < 300].groupby("PatientID")["HR"].mean().mean()
    hr_l = df[df["Time"] > tmax - 300].groupby("PatientID")["HR"].mean().mean()
    print(f"  Mean HR, first 5 min -> last 5 min : {hr_e:.1f} -> {hr_l:.1f} bpm")

    out = {"patients": int(df["PatientID"].nunique()), "ticks": int(tmax + 1),
           "dosed_patients": int(len(dosed)), "weaned_below_25pct": weaned,
           "fully_weaned": zeroed, "hr_early": round(float(hr_e), 1),
           "hr_late": round(float(hr_l), 1)}

    if "SVR0" in df.columns and df["SVR0"].notna().any():
        s_e = df[df["Time"] < 300].groupby("PatientID")["SVR0"].mean().mean()
        s_l = df[df["Time"] > tmax - 300].groupby("PatientID")["SVR0"].mean().mean()
        out["svr0_early"] = round(float(s_e), 1)
        out["svr0_late"] = round(float(s_l), 1)
        print(f"  Mean SVR0, first 5 min -> last 5 min: {s_e:.0f} -> {s_l:.0f} dyn.s/cm5")
    else:
        print("  SVR0 not recorded (optional edit not applied).")

    RESULTS["weaning"] = out


# ----------------------------------------------------------------------
def main():
    hr("SMART INFUSION AI - FULL EVALUATION")

    if not os.path.exists(TRAIN_CSV):
        sys.exit(f"Missing {TRAIN_CSV}. Run from your project root.")

    test_path = TEST_CSV if os.path.exists(TEST_CSV) else FALLBACK_TEST
    if not os.path.exists(test_path):
        sys.exit("No test CSV found.")
    if test_path == FALLBACK_TEST:
        print(f"\n  NOTE: using the small 60-patient set ({FALLBACK_TEST}).")

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(test_path)
    print(f"\n  Train: {TRAIN_CSV}  ({len(train):,} rows)")
    print(f"  Test : {test_path}  ({len(test):,} rows)")
    RESULTS["test_file"] = test_path

    section_population(train, test)
    model, y_pred, y_true = section_action(train, test)
    section_flow(train, test)
    section_risk(train, test)
    if y_pred is not None:
        section_confusion(test, y_pred, y_true)
    section_weaning()

    hr("COPY EVERYTHING BELOW INTO THE CHAT")
    print(json.dumps(RESULTS, indent=2))


if __name__ == "__main__":
    main()
