"""
Feature Extraction Module for Inbox Triage Agent.
Implements the hybrid feature extraction pipeline:
  - Subject Line: TF-IDF
  - Email Body: Bag of Words (CountVectorizer)
  - Sender Information: Domain & Interaction History Score
  - Time Information: Received Time / Hour of Day
  - Metadata: Links and Attachments presence
"""

import re
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MaxAbsScaler
from src.preprocessing import clean_text, extract_url_count


KNOWN_CORPORATE_DOMAINS = {"company.com", "corp.org", "enterprise.io", "techcorp.net", "university.edu", "acme.corp"}
KNOWN_WEBMAIL_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"}
KNOWN_SPAM_DOMAINS = {"prizewinners.xyz", "crypto-rewards.top", "urgent-notice.biz", "free-loans.ru", "lottery-draw.info"}


def parse_sender_domain(sender: str) -> str:
    """Extracts domain from email address (e.g., 'boss@company.com' -> 'company.com')."""
    if not isinstance(sender, str):
        return "unknown"
    match = re.search(r"@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", sender)
    if match:
        return match.group(1).lower()
    return "unknown"


def categorize_domain(domain: str) -> str:
    """Categorizes domain into corporate, webmail, spam_suspicious, or general."""
    domain = domain.lower()
    if domain in KNOWN_CORPORATE_DOMAINS:
        return "corporate"
    if domain in KNOWN_WEBMAIL_DOMAINS:
        return "webmail"
    if domain in KNOWN_SPAM_DOMAINS or any(domain.endswith(tld) for tld in [".xyz", ".top", ".biz", ".ru", ".info"]):
        return "spam_suspicious"
    return "general"


def parse_time_of_day(time_str: Any) -> float:
    """
    Parses time into hour of day (0.0 to 23.99).
    Accepts:
      - String HH:MM or HH:MM:SS (e.g. "14:30")
      - Integer/Float hour (e.g. 14 or 14.5)
      - Datetime/Timestamp object
    """
    if pd.isna(time_str):
        return 12.0
    if isinstance(time_str, (int, float)):
        return float(time_str) % 24.0
    if hasattr(time_str, "hour"):
        return float(time_str.hour + (time_str.minute / 60.0))

    s = str(time_str).strip()
    match = re.search(r"(\d{1,2}):(\d{2})", s)
    if match:
        h, m = int(match.group(1)), int(match.group(2))
        return float(h + (m / 60.0))
    try:
        return float(s) % 24.0
    except ValueError:
        return 12.0


class EmailFeaturePipeline:
    """
    Comprehensive feature extraction pipeline adhering to Section 7 & 10:
      - Subject -> TF-IDF (TfidfVectorizer)
      - Body -> Bag of Words (CountVectorizer)
      - Metadata -> Domain type, Sender history, Hour of Day, Has Links, Has Attachments
    """

    def __init__(self, max_subject_features: int = 500, max_body_features: int = 1500):
        self.max_subject_features = max_subject_features
        self.max_body_features = max_body_features

        # Subject feature extractor: TF-IDF
        self.subject_vectorizer = TfidfVectorizer(
            max_features=max_subject_features,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        # Body feature extractor: Bag of Words (CountVectorizer)
        self.body_vectorizer = CountVectorizer(
            max_features=max_body_features,
            ngram_range=(1, 2)
        )

        # Metadata encoders
        self.domain_encoder = OneHotEncoder(
            categories=[["corporate", "webmail", "spam_suspicious", "general"]],
            handle_unknown="ignore",
            sparse_output=False
        )
        self.scaler = MinMaxScaler()

        self.body_scaler = MaxAbsScaler()
        self.is_fitted = False

    def _prepare_metadata_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts and formats numeric and categorical metadata arrays."""
        # Ensure required columns exist with defaults
        if "sender" in df:
            domains = df["sender"].apply(parse_sender_domain)
        elif "sender_domain" in df:
            domains = df["sender_domain"]
        else:
            domains = pd.Series(["general"] * len(df))

        domain_categories = np.asarray(domains.apply(categorize_domain)).reshape(-1, 1)

        # Sender history: 0.0 (unseen) to 1.0 (frequent partner/boss)
        if "sender_history_score" in df:
            history = np.asarray(df["sender_history_score"].fillna(0.3).astype(float))
        else:
            history = np.full(len(df), 0.3)

        # Time of day: hours normalized
        if "received_time" in df:
            hours = np.asarray(df["received_time"].apply(parse_time_of_day))
        elif "hour_of_day" in df:
            hours = np.asarray(df["hour_of_day"].astype(float))
        else:
            hours = np.full(len(df), 12.0)

        # Binary links and attachments
        if "has_links" in df:
            links = np.asarray(df["has_links"].astype(int))
        else:
            links = np.zeros(len(df))

        if "has_attachments" in df:
            attachments = np.asarray(df["has_attachments"].astype(int))
        else:
            attachments = np.zeros(len(df))

        numeric_meta = np.column_stack([hours, history, links, attachments])
        return domain_categories, numeric_meta

    def fit(self, df: pd.DataFrame):
        """Fits TF-IDF on subject, CountVectorizer on body, and encoders on metadata."""
        # Preprocess text
        cleaned_subjects = df["subject"].fillna("").apply(clean_text)
        cleaned_bodies = df["body"].fillna("").apply(clean_text)

        self.subject_vectorizer.fit(cleaned_subjects)
        raw_body = self.body_vectorizer.fit_transform(cleaned_bodies)
        self.body_scaler.fit(raw_body)

        domain_cat, numeric_meta = self._prepare_metadata_df(df)
        self.domain_encoder.fit(domain_cat)
        self.scaler.fit(numeric_meta)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, feature_subset: str = "all") -> csr_matrix:
        """
        Transforms email data into feature matrix.
        feature_subset options for ablation studies:
          - 'all': Subject TF-IDF + Body BoW + Metadata
          - 'text_only': Subject TF-IDF + Body BoW
          - 'subject_only': Subject TF-IDF
          - 'body_only': Body BoW
          - 'metadata_only': Domain + Time + Links + Attachments
        """
        if not self.is_fitted:
            raise ValueError("FeaturePipeline must be fitted before calling transform().")

        cleaned_subjects = df["subject"].fillna("").apply(clean_text)
        cleaned_bodies = df["body"].fillna("").apply(clean_text)

        subj_feat = self.subject_vectorizer.transform(cleaned_subjects)
        raw_body_feat = self.body_vectorizer.transform(cleaned_bodies)
        body_feat = self.body_scaler.transform(raw_body_feat)

        domain_cat, numeric_meta = self._prepare_metadata_df(df)
        domain_feat = self.domain_encoder.transform(domain_cat)
        scaled_numeric = self.scaler.transform(numeric_meta)
        meta_feat = csr_matrix(np.hstack([domain_feat, scaled_numeric]))

        if feature_subset == "subject_only":
            return subj_feat
        elif feature_subset == "body_only":
            return body_feat
        elif feature_subset == "text_only":
            return hstack([subj_feat, body_feat], format="csr")
        elif feature_subset == "metadata_only":
            return meta_feat
        else: # 'all'
            return hstack([subj_feat, body_feat, meta_feat], format="csr")

    def fit_transform(self, df: pd.DataFrame, feature_subset: str = "all") -> csr_matrix:
        """Fits on df and transforms it."""
        self.fit(df)
        return self.transform(df, feature_subset=feature_subset)

    def get_feature_names(self) -> List[str]:
        """Returns ordered list of all feature names for interpretability."""
        subj_names = [f"subj_tfidf_{n}" for n in self.subject_vectorizer.get_feature_names_out()]
        body_names = [f"body_bow_{n}" for n in self.body_vectorizer.get_feature_names_out()]
        domain_names = [f"domain_{c}" for c in self.domain_encoder.get_feature_names_out()]
        meta_names = ["meta_hour_of_day", "meta_sender_history", "meta_has_links", "meta_has_attachments"]
        return subj_names + body_names + domain_names + meta_names
