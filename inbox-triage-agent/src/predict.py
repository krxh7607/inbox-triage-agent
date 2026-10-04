"""
Prediction and Inference Module for Inbox Triage Agent.
Provides unified prediction interface for single emails or batches of emails,
supporting Naive Bayes, Logistic Regression, and SVM (RBF).
"""

import os
from typing import Dict, Any, List, Union
import joblib
import pandas as pd
import numpy as np

from src.features import EmailFeaturePipeline
from src.agent import AgentActionEngine
from src.evaluation import LABELS_ORDER


class EmailClassifierPredictor:
    """
    Inference engine that wraps trained models, feature pipeline, and agent layer.
    """

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir

        self.pipeline_path = os.path.join(base_dir, "models", "feature_pipeline.joblib")
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Feature pipeline not found at {self.pipeline_path}. Run train.py first.")

        self.pipeline: EmailFeaturePipeline = joblib.load(self.pipeline_path)

        # Load models
        self.models = {}
        model_paths = {
            "Naive Bayes": os.path.join(base_dir, "models", "naive_bayes", "model.joblib"),
            "Logistic Regression": os.path.join(base_dir, "models", "logistic_regression", "model.joblib"),
            "SVM (RBF)": os.path.join(base_dir, "models", "svm", "model.joblib")
        }

        for name, path in model_paths.items():
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
            else:
                print(f"Warning: Model file for {name} not found at {path}")

        self.agent_engine = AgentActionEngine()

    def predict_single(
        self,
        email_data: Dict[str, Any],
        model_name: str = "SVM (RBF)",
        execute_agent: bool = True
    ) -> Dict[str, Any]:
        """
        Classifies a single email and optionally triggers the Agent Action Layer.
        """
        if model_name not in self.models:
            fallback = list(self.models.keys())[0] if self.models else None
            if not fallback:
                raise ValueError("No trained models are loaded.")
            model_name = fallback

        clf = self.models[model_name]
        df_single = pd.DataFrame([email_data])

        X = self.pipeline.transform(df_single, feature_subset="all")
        pred_label = clf.predict(X)[0]

        # Calculate prediction probabilities if supported
        probabilities = {}
        confidence = 1.0
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[0]
            classes = clf.classes_
            probabilities = {cls: float(p) for cls, p in zip(classes, probs)}
            confidence = float(probabilities.get(pred_label, 1.0))
        elif hasattr(clf, "decision_function"):
            decision = clf.decision_function(X)[0]
            # Softmax approximation
            exp_dec = np.exp(decision - np.max(decision))
            probs = exp_dec / np.sum(exp_dec)
            classes = clf.classes_
            probabilities = {cls: float(p) for cls, p in zip(classes, probs)}
            confidence = float(probabilities.get(pred_label, 1.0))

        result = {
            "model_used": model_name,
            "predicted_category": pred_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "email_input": email_data
        }

        if execute_agent:
            action_result = self.agent_engine.execute_action(
                email=email_data,
                category=pred_label,
                confidence=confidence
            )
            result["agent_action"] = action_result

        return result

    def predict_batch(
        self,
        df_emails: pd.DataFrame,
        model_name: str = "SVM (RBF)",
        execute_agent: bool = True
    ) -> pd.DataFrame:
        """
        Classifies a batch of emails and appends triage and action columns.
        """
        if model_name not in self.models:
            model_name = list(self.models.keys())[0]

        clf = self.models[model_name]
        X = self.pipeline.transform(df_emails, feature_subset="all")
        preds = clf.predict(X)

        df_result = df_emails.copy()
        df_result["predicted_category"] = preds
        df_result["model_used"] = model_name

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)
            classes = list(clf.classes_)
            confidences = [probs[i, classes.index(pred)] for i, pred in enumerate(preds)]
            df_result["confidence"] = confidences
        else:
            df_result["confidence"] = 1.0

        if execute_agent:
            actions = []
            action_details = []
            for idx, row in df_result.iterrows():
                email_dict = row.to_dict()
                cat = row["predicted_category"]
                conf = row.get("confidence", 1.0)
                act = self.agent_engine.execute_action(email_dict, cat, conf)
                actions.append(act["action_description"])
                action_details.append(act["details"])
            df_result["agent_action"] = actions
            df_result["action_details"] = action_details

        return df_result


if __name__ == "__main__":
    predictor = EmailClassifierPredictor()
    sample = {
        "email_id": "TEST-01",
        "sender": "boss@company.com",
        "subject": "CRITICAL: Database outage in production cluster",
        "body": "The primary database is unreachable. All engineers join war room immediately.",
        "received_time": "14:30",
        "has_links": 0,
        "has_attachments": 0
    }
    for m in ["Naive Bayes", "Logistic Regression", "SVM (RBF)"]:
        res = predictor.predict_single(sample, model_name=m)
        print(f"{m} -> Category: {res['predicted_category']} (Conf: {res['confidence']:.1%})")
        print(f"Action: {res['agent_action']['action_description']}")
