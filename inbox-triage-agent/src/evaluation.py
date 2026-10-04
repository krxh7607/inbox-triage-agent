"""
Model Evaluation Module for Inbox Triage Agent.
Computes and visualizes:
  - Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score (Macro/Weighted)
  - Confusion Matrix (raw and normalized)
  - Per-class classification reports for all 5 labels
  - Comparison against Presentation Reference Targets (NB: 78%, LR: 85%, SVM: 88%)
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

LABELS_ORDER = ["URGENT", "NEEDS-REPLY", "FYI", "SPAM", "DELEGATE"]

PRESENTATION_REFERENCES = {
    "Naive Bayes": 0.78,
    "Logistic Regression": 0.85,
    "SVM (RBF)": 0.88
}


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for multi-class classification.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, labels=LABELS_ORDER, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=LABELS_ORDER, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=LABELS_ORDER, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, labels=LABELS_ORDER, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, labels=LABELS_ORDER, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=LABELS_ORDER, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_true, y_pred, labels=LABELS_ORDER, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDER)

    ref_acc = PRESENTATION_REFERENCES.get(model_name, None)
    diff_from_ref = (acc - ref_acc) if ref_acc is not None else None

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "presentation_reference": ref_acc,
        "diff_from_reference": float(diff_from_ref) if diff_from_ref is not None else None,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "labels": LABELS_ORDER
    }


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: str,
    normalize: bool = False
):
    """
    Renders and saves confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 5.5))

    plot_data = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] if normalize else cm
    fmt = ".2f" if normalize else "d"

    sns.heatmap(
        plot_data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=LABELS_ORDER,
        yticklabels=LABELS_ORDER,
        cbar=True
    )
    title = f"Confusion Matrix - {model_name}" + (" (Normalized)" if normalize else "")
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Category", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Category", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_comparison_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Creates a summary DataFrame comparing models with presentation targets.
    """
    records = []
    for r in results_list:
        ref_str = f"{r['presentation_reference'] * 100:.1f}%" if r['presentation_reference'] else "N/A"
        records.append({
            "Model": r["model_name"],
            "Accuracy": f"{r['accuracy'] * 100:.2f}%",
            "Macro F1": f"{r['f1_macro'] * 100:.2f}%",
            "Weighted F1": f"{r['f1_weighted'] * 100:.2f}%",
            "Precision (Macro)": f"{r['precision_macro'] * 100:.2f}%",
            "Recall (Macro)": f"{r['recall_macro'] * 100:.2f}%",
            "PPT Reference": ref_str,
            "Ref Difference": f"{r['diff_from_reference'] * 100:+.2f}%" if r['diff_from_reference'] is not None else "N/A"
        })
    return pd.DataFrame(records)
