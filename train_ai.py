"""
===============================================================================
TRAIN_AI.PY
Smart AI Infusion Pump – Clinical Decision Support Model Training

Purpose
-------
This module trains the machine learning models responsible for
recommending infusion pump adjustments.

Two independent supervised learning models are developed using the
mechanistic digital twin dataset:

    1. Action Classification Model
       Predicts the recommended clinical action:

           • Increase infusion
           • Maintain current infusion
           • Decrease infusion
           • Emergency intervention

    2. Flow Rate Regression Model
       Predicts the DELTA between the expert controller's recommended
       flow rate and the current flow rate (RecommendedFlow - FlowRate),
       not the absolute flow rate itself.

Why delta, not absolute flow
-------------------------------
An earlier version of this script predicted RecommendedFlow directly
and reported R^2=1.000, MAE=0.180 mL/hr - which looked excellent but
was misleading: feature importance showed the model relied on FlowRate
(the CURRENT rate, also an input feature) for 99.9% of its decision.
Because expert_controller.py only moves the rate by a small amount per
tick, "predict output ~= current input" is already almost exactly
right, so the near-perfect score reflected the controller's own
smoothness, not genuine learned decision logic - the exact same issue
train_hypertensive_ai.py already found and fixed on the hypertensive
side. Predicting the delta instead removes the problem: the model can
no longer just copy an input to score well. The absolute flow
prediction (for clinically-meaningful reporting, and for
native_app.py's live AI-flow display) is reconstructed afterward as
FlowRate + predicted_delta and reported alongside the delta metrics -
a real, honest, non-inflated number this way.

Both models are evaluated exclusively using an independent test
population generated with a different simulation seed to ensure an
unbiased assessment of generalization performance.

Model Inputs
------------
Only physiologically measurable bedside variables are provided to the
models.

Variables unavailable to a real infusion pump (such as systemic
vascular resistance, cardiac output, plasma drug concentration and drug
effect) are intentionally excluded.

Similarly, simulation-specific variables that would effectively reveal
the correct answer (for example clinical profile and SAPS II score) are
also excluded.

The rationale for every included and excluded feature is documented in
generate_dataset.py.

===============================================================================
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)

from generate_dataset import FEATURE_COLUMNS, LABEL_COLUMN

CLASS_LABEL_COLUMN = "AI_Action"
CLASS_ORDER = ["Maintain", "Increase", "Decrease", "Emergency"]

# [ASSUMPTION] Small max_depth grid for validation-based model selection,
# mirroring train_hypertensive_ai.py's identical pattern - see that
# module for the full rationale. The septic-side dataset (210 train
# patients x 600 ticks = 126,000 rows) is small enough that no training
# subsample is needed here, unlike the hypertensive side's 2.52M-row set.
DEPTH_GRID = [10, 15, 20]

# Deliberately shallower than the classifier's grid - see module
# docstring "Why delta, not absolute flow". A delta target is a much
# lower-cardinality, more compressible signal than the near-continuous
# absolute flow value was, so it doesn't need the same depth budget.
REGRESSOR_DEPTH_GRID = [6, 10, 14]

SEARCH_N_ESTIMATORS = 150
FINAL_N_ESTIMATORS = 500

MODELS_DIR = "models"
FIGURES_DIR = "figures"
DATA_DIR = "data"

DELTA_LABEL_COLUMN = "FlowDelta"


def _add_delta_column(df):
    # RecommendedFlow (LABEL_COLUMN) and FlowRate already exist in the
    # generated CSVs - no dataset regeneration needed to switch the
    # regression target, just a derived column computed at load time.
    df[DELTA_LABEL_COLUMN] = df[LABEL_COLUMN] - df["FlowRate"]
    return df


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_patients.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_patients.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_patients.csv"))
    for df in (train_df, val_df, test_df):
        _add_delta_column(df)
    return train_df, val_df, test_df


def select_best_depth_classifier(train_df, val_df):
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[CLASS_LABEL_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[CLASS_LABEL_COLUMN]

    print(f"\n=== CLASSIFIER: validation-based max_depth selection (train: {len(train_df)} rows) ===")
    best_depth, best_val_acc = None, -1.0
    for depth in DEPTH_GRID:
        t0 = time.time()
        clf = RandomForestClassifier(
            n_estimators=SEARCH_N_ESTIMATORS, max_depth=depth,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        print(f"  max_depth={depth:3d}  val_accuracy={val_acc:.4f}  ({time.time()-t0:.1f}s)")
        if val_acc > best_val_acc:
            best_val_acc, best_depth = val_acc, depth

    print(f"Selected max_depth={best_depth} (val_accuracy={best_val_acc:.4f})")
    return best_depth


def select_best_depth_regressor(train_df, val_df):
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[DELTA_LABEL_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[DELTA_LABEL_COLUMN]

    print(f"\n=== REGRESSOR (delta target): validation-based max_depth selection "
          f"(train: {len(train_df)} rows) ===")
    best_depth, best_val_mae = None, float("inf")
    for depth in REGRESSOR_DEPTH_GRID:
        t0 = time.time()
        reg = RandomForestRegressor(
            n_estimators=SEARCH_N_ESTIMATORS, max_depth=depth,
            random_state=42, n_jobs=-1
        )
        reg.fit(X_train, y_train)
        val_mae = mean_absolute_error(y_val, reg.predict(X_val))
        print(f"  max_depth={depth:3d}  val_delta_MAE={val_mae:.4f} mL/hr  ({time.time()-t0:.1f}s)")
        if val_mae < best_val_mae:
            best_val_mae, best_depth = val_mae, depth

    print(f"Selected max_depth={best_depth} (val_delta_MAE={best_val_mae:.4f})")
    return best_depth


def train_final_classifier(train_df, val_df, max_depth):
    # Refit on TRAIN + VALIDATION for the final model - validation has
    # already done its job (picking max_depth); holding it out of the
    # final fit any longer would just discard 30 patients' worth of
    # real data for no remaining benefit. TEST stays untouched.
    combined = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Final classifier fit rows: {len(combined)}")
    X, y = combined[FEATURE_COLUMNS], combined[CLASS_LABEL_COLUMN]

    clf = RandomForestClassifier(
        n_estimators=FINAL_N_ESTIMATORS, max_depth=max_depth,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X, y)
    return clf


def train_final_regressor(train_df, val_df, max_depth):
    combined = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Final regressor fit rows: {len(combined)}")
    X, y = combined[FEATURE_COLUMNS], combined[DELTA_LABEL_COLUMN]

    reg = RandomForestRegressor(
        n_estimators=FINAL_N_ESTIMATORS, max_depth=max_depth,
        random_state=42, n_jobs=-1
    )
    reg.fit(X, y)
    return reg


def evaluate_classifier(clf, test_df):
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[CLASS_LABEL_COLUMN]
    y_pred = clf.predict(X_test)  # returns readable strings directly

    overall_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)

    print("\n=== CLASSIFIER: held-out test performance ===")
    print(f"Overall accuracy: {overall_acc:.3f}")
    print(classification_report(y_test, y_pred, labels=CLASS_ORDER, zero_division=0))

    # Per-profile accuracy - the real generalization claim
    test_df = test_df.copy()
    test_df["Pred_Action"] = y_pred
    per_profile = test_df.groupby("Profile").apply(
        lambda g: accuracy_score(g[CLASS_LABEL_COLUMN], g["Pred_Action"])
    )
    print("Accuracy by Profile:")
    print(per_profile.round(3))

    # Confusion matrix plot - plain matplotlib, no seaborn
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Classifier Confusion Matrix (held-out test set)")

    thresh = cm.max() / 2
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "classifier_confusion_matrix.png"), dpi=150)
    plt.close()

    return {
        "overall_accuracy": round(overall_acc, 4),
        "per_class": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                      for k, v in report.items() if k in CLASS_ORDER},
        "per_profile_accuracy": {k: round(v, 4) for k, v in per_profile.to_dict().items()}
    }


def evaluate_regressor(reg, test_df):
    X_test = test_df[FEATURE_COLUMNS]
    y_test_delta = test_df[DELTA_LABEL_COLUMN]
    y_pred_delta = reg.predict(X_test)

    # Delta-scale metrics - the honest test of whether the model
    # learned anything beyond "copy the current rate". The natural
    # baseline to compare against is "predict zero change", not zero
    # itself - R^2 already does this (it's relative to the mean delta,
    # not to zero), but printing the naive-baseline MAE alongside makes
    # the comparison legible without having to know that.
    r2_delta = r2_score(y_test_delta, y_pred_delta)
    mae_delta = mean_absolute_error(y_test_delta, y_pred_delta)
    rmse_delta = np.sqrt(mean_squared_error(y_test_delta, y_pred_delta))
    naive_zero_change_mae = float(y_test_delta.abs().mean())

    print("\n=== REGRESSOR (delta target): held-out test performance ===")
    print(f"Delta R^2:   {r2_delta:.3f}")
    print(f"Delta MAE:   {mae_delta:.4f} mL/hr")
    print(f"Delta RMSE:  {rmse_delta:.4f} mL/hr")
    print(f"'Predict zero change' baseline MAE: {naive_zero_change_mae:.4f} mL/hr "
          f"(what a model with NO learned skill would score)")

    # Reconstructed absolute-flow numbers, for clinically-meaningful
    # reporting - NOT what the model was optimized against, just a
    # derived view (FlowRate + predicted_delta) of the same predictions.
    y_test_abs = test_df[LABEL_COLUMN]
    y_pred_abs = test_df["FlowRate"].values + y_pred_delta
    mae_abs = mean_absolute_error(y_test_abs, y_pred_abs)
    rmse_abs = np.sqrt(mean_squared_error(y_test_abs, y_pred_abs))
    r2_abs = r2_score(y_test_abs, y_pred_abs)
    print(f"\nReconstructed absolute flow (FlowRate + predicted delta):")
    print(f"  R^2:  {r2_abs:.4f}   MAE: {mae_abs:.4f} mL/hr   RMSE: {rmse_abs:.4f} mL/hr")

    test_df = test_df.copy()
    test_df["Pred_Flow"] = y_pred_abs
    test_df["AbsError"] = (test_df["Pred_Flow"] - test_df[LABEL_COLUMN]).abs()
    per_profile_mae = test_df.groupby("Profile")["AbsError"].mean()
    print("Reconstructed absolute-flow MAE by Profile:")
    print(per_profile_mae.round(3))

    return {
        "delta_r2": round(float(r2_delta), 4),
        "delta_mae": round(float(mae_delta), 4),
        "delta_rmse": round(float(rmse_delta), 4),
        "naive_zero_change_mae": round(naive_zero_change_mae, 4),
        "reconstructed_absolute_flow": {
            "r2": round(float(r2_abs), 4),
            "mae": round(float(mae_abs), 4),
            "rmse": round(float(rmse_abs), 4),
        },
        "per_profile_mae": {k: round(v, 4) for k, v in per_profile_mae.to_dict().items()}
    }


def print_and_plot_feature_importance(model, model_name, filename):
    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print(f"\n=== {model_name} feature importance ===")
    print(importance.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.barh(importance["Feature"][::-1], importance["Importance"][::-1])
    plt.title(f"{model_name} Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.close()

    return importance


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train_df, val_df, test_df = load_data()
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")
    print(f"Features used: {FEATURE_COLUMNS}")

    best_clf_depth = select_best_depth_classifier(train_df, val_df)
    best_reg_depth = select_best_depth_regressor(train_df, val_df)

    clf = train_final_classifier(train_df, val_df, best_clf_depth)
    reg = train_final_regressor(train_df, val_df, best_reg_depth)

    clf_metrics = evaluate_classifier(clf, test_df)
    reg_metrics = evaluate_regressor(reg, test_df)

    print_and_plot_feature_importance(clf, "Classifier", "classifier_feature_importance.png")
    print_and_plot_feature_importance(reg, "Regressor", "regressor_feature_importance.png")

    joblib.dump(clf, os.path.join(MODELS_DIR, "ai_action_classifier.pkl"))
    joblib.dump(reg, os.path.join(MODELS_DIR, "ai_flow_regressor.pkl"))

    metrics = {
        "classifier": clf_metrics, "regressor": reg_metrics, "features_used": FEATURE_COLUMNS,
        "selected_max_depth": {"classifier": best_clf_depth, "regressor": best_reg_depth}
    }
    with open(os.path.join(FIGURES_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModels saved to {MODELS_DIR}/: ai_action_classifier.pkl, ai_flow_regressor.pkl")
    print(f"Metrics + plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
