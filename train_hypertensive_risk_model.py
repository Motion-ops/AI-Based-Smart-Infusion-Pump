"""
===============================================================================
TRAIN_HYPERTENSIVE_RISK_MODEL.PY
Smart AI Infusion Pump - Hypertensive Emergency / Sodium Nitroprusside

Purpose
-------
Sibling of train_risk_model.py. Trains the early-warning model for this
mode: given only the current and recent-past bedside measurements
(never the future), predict WillBreachFloor - whether MAP is about to
fall below its own staged safety floor, or the cyanide rate-risk flag
is about to trip, within the next 30 seconds (see
generate_hypertensive_dataset.py for exactly how that label is built
from each patient's own future trajectory, and why it can only be
computed in hindsight during dataset generation, never at prediction
time).

This is the hypertensive-mode counterpart to the septic side's
WillDeteriorate risk model, watching for the OPPOSITE danger:
overcorrecting blood pressure too fast, rather than under-treating it.

Same validation-based model selection as train_hypertensive_ai.py
---------------------------------------------------------------------
train_risk_model.py picks max_depth=15 without comparing it to
anything. This file instead selects max_depth by ROC-AUC on the
VALIDATION set (100 patients, seed=777, independent of both train and
test) - accuracy alone would be misleading here since WillBreachFloor
is a rare positive class (~1-2% of rows), so a model selection metric
that's actually sensitive to how well the classes are separated
matters. TEST (200 patients, seed=999) is touched exactly once, for
the final reported numbers.

Training data is subsampled for the same wall-clock reason documented
in train_hypertensive_ai.py: a full RandomForest fit on millions of
near-identical consecutive one-second ticks is needless and, at this
project's actual dataset size, was empirically catastrophic (a single
fit once ran over 5 hours without finishing). Validation and test stay
full-size.
===============================================================================
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

from generate_hypertensive_dataset import RISK_FEATURE_COLUMNS, RISK_LABEL_COLUMN, LOOKAHEAD

DEPTH_GRID = [10, 15, 20]
SEARCH_N_ESTIMATORS = 150
FINAL_N_ESTIMATORS = 500

# Same reasoning as train_hypertensive_ai.py's _resolve_subsample_fraction
# - re-derived from the actual row count rather than a fixed fraction,
# so this stays fast regardless of how long a future dataset
# regeneration decides to simulate.
_TARGET_SUBSAMPLE_ROWS = 150_000
SUBSAMPLE_SEED = 42

MODELS_DIR = "models"
FIGURES_DIR = "figures"
DATA_DIR = "data"


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "hypertensive_train_patients.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "hypertensive_val_patients.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "hypertensive_test_patients.csv"))
    return train_df, val_df, test_df


def _subsample(df, seed=SUBSAMPLE_SEED):
    frac = min(1.0, _TARGET_SUBSAMPLE_ROWS / len(df))
    return df.sample(frac=frac, random_state=seed)


def select_best_depth(train_df, val_df):
    train_sub = _subsample(train_df)
    X_train, y_train = train_sub[RISK_FEATURE_COLUMNS], train_sub[RISK_LABEL_COLUMN]
    X_val, y_val = val_df[RISK_FEATURE_COLUMNS], val_df[RISK_LABEL_COLUMN]

    print(f"\n=== RISK MODEL: validation-based max_depth selection "
          f"(train subsample: {len(train_sub)} rows, positive rate={y_train.mean():.4f}) ===")
    best_depth, best_val_auc = None, -1.0
    for depth in DEPTH_GRID:
        t0 = time.time()
        clf = RandomForestClassifier(
            n_estimators=SEARCH_N_ESTIMATORS, max_depth=depth,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        val_proba = clf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        val_recall = recall_score(y_val, clf.predict(X_val), zero_division=0)
        print(f"  max_depth={depth:3d}  val_ROC_AUC={val_auc:.4f}  "
              f"val_recall={val_recall:.4f}  ({time.time()-t0:.1f}s)")
        if val_auc > best_val_auc:
            best_val_auc, best_depth = val_auc, depth

    print(f"Selected max_depth={best_depth} (val_ROC_AUC={best_val_auc:.4f})")
    return best_depth


def train_final_risk_classifier(train_df, val_df, max_depth):
    # Refit on subsampled-TRAIN + full VALIDATION, same convention as
    # train_hypertensive_ai.py - validation has already done its job
    # (picking max_depth), so folding it into the final fit uses 100
    # more real patients rather than discarding them. TEST stays
    # completely untouched either way.
    combined = pd.concat([_subsample(train_df), val_df], ignore_index=True)
    print(f"Final risk classifier fit rows: {len(combined)} "
          f"(positive rate={combined[RISK_LABEL_COLUMN].mean():.4f})")

    X, y = combined[RISK_FEATURE_COLUMNS], combined[RISK_LABEL_COLUMN]

    clf = RandomForestClassifier(
        n_estimators=FINAL_N_ESTIMATORS, max_depth=max_depth,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def evaluate_risk_classifier(clf, test_df):
    X_test = test_df[RISK_FEATURE_COLUMNS]
    y_test = test_df[RISK_LABEL_COLUMN]

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print("\n=== RISK MODEL: held-out TEST performance (touched once) ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    test_df = test_df.copy()
    test_df["Pred"] = y_pred
    per_profile_recall = (
        test_df[test_df[RISK_LABEL_COLUMN] == 1]
        .groupby("Profile")
        .apply(lambda g: recall_score(g[RISK_LABEL_COLUMN], g["Pred"], zero_division=0))
    )
    print("\nRecall by Clinical Profile (among true positives only):")
    print(per_profile_recall.round(3))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Hypertensive Early Breach-Risk Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hypertensive_risk_model_roc_curve.png"), dpi=150)
    plt.close()

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    labels = ["No Risk", "Breach Risk"]
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Reds")
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Hypertensive Risk Model Confusion Matrix")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hypertensive_risk_model_confusion_matrix.png"), dpi=150)
    plt.close()

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "recall_by_profile": {k: round(v, 4) for k, v in per_profile_recall.to_dict().items()},
        "lookahead_ticks": LOOKAHEAD,
    }


def print_and_plot_feature_importance(clf):
    importance = pd.DataFrame({
        "Feature": RISK_FEATURE_COLUMNS,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print("\n=== Hypertensive Risk Model Feature Importance ===")
    print(importance.to_string(index=False))

    plt.figure(figsize=(8, 6))
    plt.barh(importance["Feature"][::-1], importance["Importance"][::-1])
    plt.title("Hypertensive Risk Model Feature Importance")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hypertensive_risk_model_feature_importance.png"), dpi=150)
    plt.close()

    return importance


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train_df, val_df, test_df = load_data()
    print(f"Train rows: {len(train_df)} | Validation rows: {len(val_df)} | Test rows: {len(test_df)}")
    print(f"Risk prediction features: {RISK_FEATURE_COLUMNS}")
    print(f"Positive class rate - Train: {train_df[RISK_LABEL_COLUMN].mean():.4f}, "
          f"Val: {val_df[RISK_LABEL_COLUMN].mean():.4f}, "
          f"Test: {test_df[RISK_LABEL_COLUMN].mean():.4f}")

    best_depth = select_best_depth(train_df, val_df)
    clf = train_final_risk_classifier(train_df, val_df, best_depth)

    metrics = evaluate_risk_classifier(clf, test_df)
    metrics["selected_max_depth"] = best_depth

    print_and_plot_feature_importance(clf)

    joblib.dump(clf, os.path.join(MODELS_DIR, "hypertensive_risk_classifier.pkl"))

    with open(os.path.join(FIGURES_DIR, "hypertensive_risk_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {MODELS_DIR}/hypertensive_risk_classifier.pkl")
    print(f"Evaluation results saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
