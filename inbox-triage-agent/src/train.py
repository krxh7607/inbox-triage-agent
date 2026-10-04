"""
Model Training & Ablation Study Script for Inbox Triage Agent.
Trains and evaluates:
  1. Naive Bayes (Presentation Ref: 78%)
  2. Logistic Regression (Presentation Ref: 85%)
  3. SVM with RBF Kernel (Presentation Ref: 88%)

Also conducts feature ablation study:
  - Subject TF-IDF Only
  - Body BoW Only
  - Text Only (Subject + Body)
  - Metadata Only (Domain, Time, Links, Attachments)
  - Full Hybrid Pipeline (Subject + Body + Metadata)
"""

import os
import json
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

from src.features import EmailFeaturePipeline
from src.evaluation import (
    evaluate_model_predictions,
    plot_and_save_confusion_matrix,
    generate_comparison_table,
    LABELS_ORDER,
    PRESENTATION_REFERENCES
)


def train_and_evaluate_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "emails_raw.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw dataset not found at {data_path}. Run generate_dataset.py first.")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df)} samples across classes: {df['label'].unique()}")

    # Stratified Train-Test Split (80% train, 20% test)
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["label"]
    )
    print(f"Train split: {len(train_df)} samples, Test split: {len(test_df)} samples.")

    # 1. Fit Feature Pipeline
    print("\n--- Fitting Hybrid Feature Extraction Pipeline ---")
    pipeline = EmailFeaturePipeline(max_subject_features=500, max_body_features=1500)
    pipeline.fit(train_df)

    X_train = pipeline.transform(train_df, feature_subset="all")
    y_train = train_df["label"].values

    X_test = pipeline.transform(test_df, feature_subset="all")
    y_test = test_df["label"].values

    # Save pipeline
    joblib.dump(pipeline, os.path.join(base_dir, "models", "feature_pipeline.joblib"))
    print("Saved feature pipeline to models/feature_pipeline.joblib")

    # Define the 3 core models specified in presentation
    models_to_train = {
        "Naive Bayes": {
            "model": MultinomialNB(alpha=0.5),
            "save_dir": os.path.join(base_dir, "models", "naive_bayes")
        },
        "Logistic Regression": {
            "model": LogisticRegression(C=1.5, max_iter=1000, random_state=42),
            "save_dir": os.path.join(base_dir, "models", "logistic_regression")
        },
        "SVM (RBF)": {
            "model": SVC(kernel="rbf", C=5.0, gamma="auto", probability=True, random_state=42),
            "save_dir": os.path.join(base_dir, "models", "svm")
        }
    }

    results = []
    print("\n--- Training & Evaluating Core ML Models ---")
    for name, config in models_to_train.items():
        print(f"\nTraining {name}...")
        clf = config["model"]
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)
        metrics = evaluate_model_predictions(y_test, y_pred, model_name=name)
        results.append(metrics)

        # Save model
        os.makedirs(config["save_dir"], exist_ok=True)
        joblib.dump(clf, os.path.join(config["save_dir"], "model.joblib"))
        print(f"Saved {name} model to {config['save_dir']}/model.joblib")

        # Save Confusion Matrix
        cm_path = os.path.join(base_dir, "outputs", "figures", f"confusion_matrix_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
        plot_and_save_confusion_matrix(np.array(metrics["confusion_matrix"]), name, cm_path)

        print(f"-> {name} Accuracy: {metrics['accuracy']*100:.2f}% (PPT Ref: {PRESENTATION_REFERENCES[name]*100:.1f}%), Macro F1: {metrics['f1_macro']*100:.2f}%")

    # Save Comparison Metrics
    df_comparison = generate_comparison_table(results)
    metrics_dir = os.path.join(base_dir, "outputs", "metrics")
    figures_dir = os.path.join(base_dir, "outputs", "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df_comparison.to_csv(os.path.join(metrics_dir, "model_comparison.csv"), index=False)
    with open(os.path.join(metrics_dir, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Summary Model Comparison ---")
    print(df_comparison.to_string(index=False))

    # Plot Accuracy vs PPT Reference
    plt.figure(figsize=(8, 4.8))
    model_names = [r["model_name"] for r in results]
    actual_accs = [r["accuracy"] * 100 for r in results]
    ref_accs = [r["presentation_reference"] * 100 for r in results]

    x = np.arange(len(model_names))
    width = 0.35

    plt.bar(x - width/2, actual_accs, width, label="Actual Achieved Accuracy", color="#2b5c8f")
    plt.bar(x + width/2, ref_accs, width, label="Presentation Reference Target", color="#e76f51", alpha=0.85)

    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Model Accuracy: Actual Measured vs. Presentation Target", fontsize=12, fontweight="bold")
    plt.xticks(x, model_names, fontsize=10, fontweight="bold")
    plt.ylim(0, 105)
    for i in range(len(model_names)):
        plt.text(i - width/2, actual_accs[i] + 1.5, f"{actual_accs[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
        plt.text(i + width/2, ref_accs[i] + 1.5, f"{ref_accs[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "models_accuracy_comparison.png"), dpi=200)
    plt.close()

    # 2. Ablation Study across Feature Subsets (Section 8, 15)
    print("\n--- Running Feature Ablation Study ---")
    subsets = ["subject_only", "body_only", "text_only", "metadata_only", "all"]
    subset_labels = {
        "subject_only": "Subject TF-IDF Only",
        "body_only": "Body BoW Only",
        "text_only": "Text Only (Subject + Body)",
        "metadata_only": "Metadata Only (Domain/Time/Links)",
        "all": "Full Pipeline (Subject + Body + Meta)"
    }

    ablation_records = []
    # Test on Logistic Regression as standard baseline for feature ablation
    for s in subsets:
        X_tr = pipeline.transform(train_df, feature_subset=s)
        X_te = pipeline.transform(test_df, feature_subset=s)

        clf = LogisticRegression(C=1.5, max_iter=1000, random_state=42)
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)

        acc = float(np.mean(preds == y_test))
        ablation_records.append({
            "Feature Subset": subset_labels[s],
            "Subset Key": s,
            "Accuracy": acc,
            "Features Count": X_tr.shape[1]
        })
        print(f"Feature Subset: {subset_labels[s]:<38} | Features: {X_tr.shape[1]:<5} | Accuracy: {acc*100:.2f}%")

    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(os.path.join(metrics_dir, "ablation_study.csv"), index=False)
    with open(os.path.join(metrics_dir, "ablation_study.json"), "w") as f:
        json.dump(ablation_records, f, indent=2)

    # Plot Ablation Study
    plt.figure(figsize=(9, 4.5))
    bars = plt.barh(
        [r["Feature Subset"] for r in ablation_records],
        [r["Accuracy"] * 100 for r in ablation_records],
        color=["#7209b7", "#4361ee", "#4cc9f0", "#f72585", "#2a9d8f"]
    )
    plt.xlabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Ablation Study: Feature Subset Impact on Classification Accuracy", fontsize=12, fontweight="bold")
    plt.xlim(0, 105)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1.0, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "ablation_study.png"), dpi=200)
    plt.close()
    print("Ablation study saved to outputs/figures/ablation_study.png")

    print("\nTraining and evaluation pipeline complete!")


if __name__ == "__main__":
    train_and_evaluate_all()
