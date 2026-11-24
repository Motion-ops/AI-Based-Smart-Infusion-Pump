"""
===============================================================================
TRAIN_RISK_MODEL.PY
Smart AI Infusion Pump - Early Deterioration Risk Prediction Model

Purpose
-------
This module trains and evaluates the artificial intelligence model
responsible for predicting impending patient deterioration.

Unlike the infusion advisory model, which learns to imitate the expert
controller's dosing decisions, this model performs an independent
clinical prediction task.

Using only information that would realistically be available from the
bedside monitor, the model predicts whether the patient's condition
will deteriorate within a predefined future prediction horizon.

Prediction task
---------------
Given the most recent rolling window of noisy physiological
measurements, the model predicts whether the patient will experience
either:

    • Mean arterial pressure below the emergency threshold, or
    • Oxygen saturation below the critical safety threshold,

within the next LOOKAHEAD simulation steps.

The deterioration labels are generated during dataset construction by
examining each patient's future physiological trajectory. Consequently,
this model learns temporal clinical patterns rather than reproducing
existing controller logic.

Evaluation strategy
-------------------
Model performance is assessed exclusively on an independently generated
test population that was never observed during training.

Because deterioration events are intentionally uncommon within the
simulation dataset, overall classification accuracy alone provides an
incomplete assessment of performance.

Clinical usefulness is therefore evaluated primarily using:

    • Recall
        Ability to identify genuine deterioration events.

    • ROC-AUC
        Ability to distinguish between low-risk and high-risk patients.

    • Precision
        Proportion of generated alerts that correspond to true
        deterioration events.

These metrics better reflect the clinical objective of identifying
patients at risk before physiological collapse occurs.

All dataset generation procedures are documented in
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from generate_dataset import (
    RISK_FEATURE_COLUMNS,
    RISK_LABEL_COLUMN,
    LOOKAHEAD,
)

# [ASSUMPTION] Small max_depth grid for validation-based model selection,
# mirroring train_hypertensive_risk_model.py's identical pattern. No
# training subsample is needed here (see train_ai.py) - the septic-side
# dataset is small enough to fit directly.
DEPTH_GRID = [10, 15, 20]
SEARCH_N_ESTIMATORS = 150
FINAL_N_ESTIMATORS = 500

MODELS_DIR = "models"
FIGURES_DIR = "figures"
DATA_DIR = "data"


def load_data():
    """
    Loads the training, validation, and independent test datasets.

    Returns
    -------
    tuple
        Training, validation, and testing dataframes.

    Explanation
    -----------
    The datasets are generated using different random seeds so that the
    validation and test populations remain completely independent of
    the training population and of each other, providing an unbiased
    basis for both hyperparameter selection (validation) and final
    performance reporting (test).
    """
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_patients.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_patients.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_patients.csv"))
    return train_df, val_df, test_df


def select_best_depth(train_df, val_df):
    """
    Selects max_depth by scoring each candidate's ROC-AUC on the held-out
    validation set, mirroring train_hypertensive_risk_model.py.
    """
    X_train, y_train = train_df[RISK_FEATURE_COLUMNS], train_df[RISK_LABEL_COLUMN]
    X_val, y_val = val_df[RISK_FEATURE_COLUMNS], val_df[RISK_LABEL_COLUMN]

    print(f"\n=== RISK MODEL: validation-based max_depth selection "
          f"(train: {len(train_df)} rows, positive rate={y_train.mean():.4f}) ===")
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
    """
    Refits on TRAIN + VALIDATION for the final model - validation has
    already done its job (picking max_depth), so folding it into the
    final fit uses 30 more real patients rather than discarding them.
    TEST stays completely untouched.
    """
    combined = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Final risk classifier fit rows: {len(combined)} "
          f"(positive rate={combined[RISK_LABEL_COLUMN].mean():.4f})")

    X, y = combined[RISK_FEATURE_COLUMNS], combined[RISK_LABEL_COLUMN]

    clf = RandomForestClassifier(
        n_estimators=FINAL_N_ESTIMATORS,
        max_depth=max_depth,

        # Assign greater importance to deterioration events during
        # training to compensate for class imbalance.
        class_weight="balanced",

        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X, y)
    return clf


def evaluate_risk_classifier(clf, test_df):
    """
    Evaluates the trained risk prediction model.

    Parameters
    ----------
    clf : RandomForestClassifier
        Trained classifier.

    test_df : DataFrame
        Independent test dataset.

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.

    Explanation
    -----------
    Performance is assessed using an entirely independent patient
    population generated with a different simulation seed.

    Multiple evaluation metrics are reported because the dataset is
    intentionally imbalanced. Particular emphasis is placed on recall
    since failing to detect a genuine deterioration event represents
    the most clinically significant error.
    """

    X_test = test_df[RISK_FEATURE_COLUMNS]
    y_test = test_df[RISK_LABEL_COLUMN]

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print("\n=== Risk Model Performance (Independent Test Population) ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")

    print(classification_report(y_test, y_pred, zero_division=0))

    # -----------------------------------------------------------------
    # Evaluate recall separately for each clinical profile.
    #
    # This provides additional insight into model performance across
    # different illness severities.
    # -----------------------------------------------------------------
    test_df = test_df.copy()
    test_df["Pred"] = y_pred

    per_profile_recall = (
        test_df[test_df[RISK_LABEL_COLUMN] == 1]
        .groupby("Profile")
        .apply(
            lambda g: recall_score(
                g[RISK_LABEL_COLUMN],
                g["Pred"],
                zero_division=0,
            )
        )
    )

    print("\nRecall by Clinical Profile")
    print(per_profile_recall.round(3))

    # -----------------------------------------------------------------
    # Receiver Operating Characteristic (ROC) curve.
    #
    # The ROC curve illustrates the classifier's ability to separate
    # low-risk from high-risk patients across all decision thresholds.
    # -----------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="grey",
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Early Deterioration Prediction")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "risk_model_roc_curve.png",
        ),
        dpi=150,
    )

    plt.close()

    # -----------------------------------------------------------------
    # Confusion matrix.
    #
    # The confusion matrix summarizes the number of correctly and
    # incorrectly classified observations.
    # -----------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    labels = ["No Risk", "Deterioration"]

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Reds")
    plt.colorbar()

    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Risk Model Confusion Matrix")

    thresh = cm.max() / 2

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "risk_model_confusion_matrix.png",
        ),
        dpi=150,
    )

    plt.close()

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "recall_by_profile": {
            k: round(v, 4)
            for k, v in per_profile_recall.to_dict().items()
        },
        "lookahead_ticks": LOOKAHEAD,
    }


def print_and_plot_feature_importance(clf):
    """
    Displays and visualizes feature importance.

    Explanation
    -----------
    Random Forest models estimate the contribution of each predictor
    to the final classification.

    Examining feature importance provides insight into which
    physiological measurements most strongly influence deterioration
    prediction within the simulated patient population.
    """

    importance = pd.DataFrame(
        {
            "Feature": RISK_FEATURE_COLUMNS,
            "Importance": clf.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    print("\n=== Feature Importance ===")
    print(importance.to_string(index=False))

    plt.figure(figsize=(8, 6))
    plt.barh(
        importance["Feature"][::-1],
        importance["Importance"][::-1],
    )

    plt.title("Risk Model Feature Importance")
    plt.xlabel("Relative Importance")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "risk_model_feature_importance.png",
        ),
        dpi=150,
    )

    plt.close()

    return importance


def main():
    """
    Executes the complete training and evaluation workflow.

    The workflow consists of:

        1. Loading the datasets.
        2. Training the classifier.
        3. Evaluating model performance.
        4. Visualizing feature importance.
        5. Saving the trained model and evaluation results.
    """

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train_df, val_df, test_df = load_data()

    print(f"Training observations  : {len(train_df)}")
    print(f"Validation observations: {len(val_df)}")
    print(f"Testing observations   : {len(test_df)}")

    print(f"Risk prediction features: {RISK_FEATURE_COLUMNS}")

    print(
        f"Positive class rate - "
        f"Train: {train_df[RISK_LABEL_COLUMN].mean():.3f}, "
        f"Val: {val_df[RISK_LABEL_COLUMN].mean():.3f}, "
        f"Test: {test_df[RISK_LABEL_COLUMN].mean():.3f}"
    )

    best_depth = select_best_depth(train_df, val_df)
    clf = train_final_risk_classifier(train_df, val_df, best_depth)

    metrics = evaluate_risk_classifier(clf, test_df)
    metrics["selected_max_depth"] = best_depth

    print_and_plot_feature_importance(clf)

    # Save the trained classifier for deployment within the
    # intelligent infusion pump framework.
    joblib.dump(
        clf,
        os.path.join(MODELS_DIR, "risk_classifier.pkl"),
    )

    # Save evaluation metrics for later analysis and reporting.
    with open(
        os.path.join(
            FIGURES_DIR,
            "risk_model_metrics.json",
        ),
        "w",
    ) as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {MODELS_DIR}/risk_classifier.pkl")
    print(f"Evaluation results saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()