"""
===============================================================================
TRAIN_HYPERTENSIVE_AI.PY
Smart AI Infusion Pump - Hypertensive Emergency / Sodium Nitroprusside

Purpose
-------
Sibling of train_ai.py. Trains the two supervised models for this
mode from generate_hypertensive_dataset.py's output:

    1. Action Classification Model
       Predicts the recommended clinical action (Maintain / Increase /
       Decrease / Emergency).

    2. Flow DELTA Regression Model
       Predicts the CHANGE in nitroprusside flow rate (mL/hr) the
       controller recommends, i.e. RecommendedFlow - FlowRate, not the
       absolute rate itself.

Why delta, not absolute flow
------------------------------
An earlier version of this script predicted RecommendedFlow directly
and reported R^2=1.000, MAE=0.019 mL/hr - which looked excellent but
was misleading: feature importance showed the model relied on FlowRate
(the CURRENT rate, also an input feature) for 99.999% of its decision.
Because hypertensive_expert_controller.py only moves the rate by a tiny
amount per tick, "predict output ~= current input" is already almost
exactly right, so the near-perfect score reflected the controller's own
smoothness, not genuine learned decision logic. It ALSO produced a
7.46 GB model file - deep trees memorizing near-identical continuous
values in a not-quite-degenerate target take an enormous number of
nodes to represent. Predicting the delta instead removes both problems
at once: it can no longer copy an input to score well, and a smaller,
noisier target needs far shallower trees to fit. The absolute flow
prediction (for clinically-meaningful reporting) is reconstructed
afterward as FlowRate + predicted_delta and reported alongside the
delta metrics - it's a real, honest, non-inflated number this way.

Both models are evaluated exactly once, at the end, on the held-out
TEST set (200 patients, a seed never touched during training or model
selection).

What's different from train_ai.py: an actual three-way split is used
------------------------------------------------------------------------
train_ai.py only has train/test - every hyperparameter (n_estimators,
max_depth) is hardcoded, chosen without ever being formally compared
against anything. This file instead uses the VALIDATION set
(100 patients, seed=777, generated independently of both train and
test) to do real model selection: a small max_depth grid is fit on
TRAIN ONLY, each candidate is scored on VALIDATION, and only the
winning configuration is ever evaluated against TEST. This is exactly
the methodology described in this project's own dataset-generation
discussion: training data teaches the model, validation data compares
candidates, test data is touched exactly once for the number that
actually gets reported.

n_estimators is kept smaller during the search phase (150, vs. the
final model's 500) purely for wall-clock time - the training set here
is 2.52M rows (700 patients x 3600 ticks), about 17x the septic side's
train_patients.csv, so refitting a 500-tree forest for every grid
candidate would be needlessly slow when the search only needs to rank
candidates relative to each other, not produce the final model itself.
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

from generate_hypertensive_dataset import FEATURE_COLUMNS, LABEL_COLUMN

CLASS_LABEL_COLUMN = "AI_Action"

# hypertensive_expert_controller.py's cyanide safety overlay can emit
# "Toxicity-Cap" (rate-risk threshold exceeded) and "Toxicity-Stop"
# (total buffer capacity exceeded). Extending the dataset's simulated
# duration to 2 hours (see generate_hypertensive_dataset.py) made
# "Toxicity-Cap" occur for real (0.5-0.8% of rows across all three
# splits) - included here now that it's genuinely represented.
# "Toxicity-Stop" still never occurs (its 10 mg/kg total buffer
# threshold is a much larger bar than the rate-risk one) - left out
# rather than silently assumed impossible; add it back if a future,
# longer-duration regeneration produces it.
CLASS_ORDER = ["Maintain", "Increase", "Decrease", "Emergency", "Toxicity-Cap"]

# [ASSUMPTION] Small max_depth grid for validation-based model
# selection - n_estimators for the SEARCH phase only, not the final
# model (see module docstring).
DEPTH_GRID = [10, 15, 20]

# The regressor's grid is deliberately shallower than the classifier's
# - see module docstring "Why delta, not absolute flow". A delta
# target is a much lower-cardinality, more compressible signal than
# the near-continuous absolute flow value was, so it doesn't need (and
# shouldn't be given) the same depth budget - that's exactly what
# produced the 7.46 GB model file last time.
REGRESSOR_DEPTH_GRID = [6, 10, 14]

SEARCH_N_ESTIMATORS = 150
FINAL_N_ESTIMATORS = 500

# [FIX, empirically forced] RandomForest fit time on the full 2.52M-row
# training set turned out to be wildly impractical - a single 150-tree
# fit ran over 5 hours of wall time without finishing. The dataset's
# size is mostly redundant, not informative: each patient contributes
# thousands of consecutive ONE-SECOND ticks that are nearly identical
# to their neighbors (the underlying MAP/flow trajectory moves slowly
# relative to a 1s step), so a non-temporal tree model gets very little
# additional signal from seeing every single tick versus a random
# subset of them. Subsampling the TRAINING data only (validation and
# test stay full-size, since scoring/predicting is cheap regardless of
# size and this is where the honestly-reported numbers come from) to
# roughly the septic side's own train_patients.csv scale (~150K rows)
# keeps this tractable without touching what gets evaluated.
#
# The fraction here is deliberately re-derived, not fixed, since the
# generator's sim_time changed (1h -> 2h) after this was first tuned,
# doubling total row count - halving the fraction keeps the ABSOLUTE
# subsample size (and therefore fit time) roughly the same regardless
# of how long a future dataset regeneration decides to simulate.
_TARGET_SUBSAMPLE_ROWS = 150_000


def _resolve_subsample_fraction(train_row_count):
    return min(1.0, _TARGET_SUBSAMPLE_ROWS / train_row_count)


SUBSAMPLE_SEED = 42

MODELS_DIR = "models"
FIGURES_DIR = "figures"
DATA_DIR = "data"


DELTA_LABEL_COLUMN = "FlowDelta"


def _add_delta_column(df):
    # Both RecommendedFlow and FlowRate already exist in the generated
    # CSVs - no dataset regeneration needed to switch the regression
    # target, just a derived column computed at load time.
    df[DELTA_LABEL_COLUMN] = df[LABEL_COLUMN] - df["FlowRate"]
    return df


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "hypertensive_train_patients.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "hypertensive_val_patients.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "hypertensive_test_patients.csv"))
    for df in (train_df, val_df, test_df):
        _add_delta_column(df)
    return train_df, val_df, test_df


def _subsample(df, seed=SUBSAMPLE_SEED):
    frac = _resolve_subsample_fraction(len(df))
    return df.sample(frac=frac, random_state=seed)


def select_best_depth_classifier(train_df, val_df):
    train_sub = _subsample(train_df)
    X_train, y_train = train_sub[FEATURE_COLUMNS], train_sub[CLASS_LABEL_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[CLASS_LABEL_COLUMN]

    print(f"\n=== CLASSIFIER: validation-based max_depth selection (train subsample: {len(train_sub)} rows) ===")
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
    train_sub = _subsample(train_df)
    X_train, y_train = train_sub[FEATURE_COLUMNS], train_sub[DELTA_LABEL_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[DELTA_LABEL_COLUMN]

    print(f"\n=== REGRESSOR (delta target): validation-based max_depth selection "
          f"(train subsample: {len(train_sub)} rows) ===")
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
    # Refit on subsampled-TRAIN + full VALIDATION for the final model -
    # the validation set has already done its job (picking max_depth)
    # and holding it out of the final fit any longer would just be
    # throwing away 100 patients' worth of real data for no remaining
    # benefit. Only the TRAIN half is subsampled (same reasoning as
    # select_best_depth_classifier - see _resolve_subsample_fraction());
    # validation is already a manageable size on its own. TEST
    # stays completely untouched either way.
    combined = pd.concat([_subsample(train_df), val_df], ignore_index=True)
    print(f"Final classifier fit rows: {len(combined)}")
    X, y = combined[FEATURE_COLUMNS], combined[CLASS_LABEL_COLUMN]

    clf = RandomForestClassifier(
        n_estimators=FINAL_N_ESTIMATORS, max_depth=max_depth,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X, y)
    return clf


def train_final_regressor(train_df, val_df, max_depth):
    combined = pd.concat([_subsample(train_df), val_df], ignore_index=True)
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
    y_pred = clf.predict(X_test)

    overall_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)

    print("\n=== CLASSIFIER: held-out TEST performance (touched once) ===")
    print(f"Overall accuracy: {overall_acc:.3f}")
    print(classification_report(y_test, y_pred, labels=CLASS_ORDER, zero_division=0))

    test_df = test_df.copy()
    test_df["Pred_Action"] = y_pred
    per_profile = test_df.groupby("Profile").apply(
        lambda g: accuracy_score(g[CLASS_LABEL_COLUMN], g["Pred_Action"])
    )
    print("Accuracy by Profile:")
    print(per_profile.round(3))

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Hypertensive Classifier Confusion Matrix (held-out test set)")

    thresh = cm.max() / 2
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hypertensive_classifier_confusion_matrix.png"), dpi=150)
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

    print("\n=== REGRESSOR (delta target): held-out TEST performance (touched once) ===")
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
        "naive_zero_change_baseline_mae": round(naive_zero_change_mae, 4),
        "reconstructed_absolute_r2": round(float(r2_abs), 4),
        "reconstructed_absolute_mae": round(float(mae_abs), 4),
        "reconstructed_absolute_rmse": round(float(rmse_abs), 4),
        "per_profile_absolute_mae": {k: round(v, 4) for k, v in per_profile_mae.to_dict().items()}
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
    print(f"Train rows: {len(train_df)} | Validation rows: {len(val_df)} | Test rows: {len(test_df)}")
    print(f"Features used: {FEATURE_COLUMNS}")

    best_clf_depth = select_best_depth_classifier(train_df, val_df)
    best_reg_depth = select_best_depth_regressor(train_df, val_df)

    clf = train_final_classifier(train_df, val_df, best_clf_depth)
    reg = train_final_regressor(train_df, val_df, best_reg_depth)

    clf_metrics = evaluate_classifier(clf, test_df)
    reg_metrics = evaluate_regressor(reg, test_df)

    print_and_plot_feature_importance(clf, "Hypertensive Classifier", "hypertensive_classifier_feature_importance.png")
    print_and_plot_feature_importance(reg, "Hypertensive Regressor", "hypertensive_regressor_feature_importance.png")

    joblib.dump(clf, os.path.join(MODELS_DIR, "hypertensive_action_classifier.pkl"))
    joblib.dump(reg, os.path.join(MODELS_DIR, "hypertensive_flow_regressor.pkl"))

    metrics = {
        "classifier": clf_metrics,
        "regressor": reg_metrics,
        "selected_max_depth": {"classifier": best_clf_depth, "regressor": best_reg_depth},
        "features_used": FEATURE_COLUMNS,
    }
    with open(os.path.join(FIGURES_DIR, "hypertensive_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModels saved to {MODELS_DIR}/: hypertensive_action_classifier.pkl, hypertensive_flow_regressor.pkl")
    print(f"Metrics + plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
