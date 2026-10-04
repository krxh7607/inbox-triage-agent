# Inbox Triage Agent

> **An ML-Powered System to Automatically Classify and Act on Your Emails**  
> *Academic Machine Learning Project --- Email Data, Classification + Agent Layer*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 1. Project Overview

Manual email management is slow, error-prone, and mentally exhausting. The **Inbox Triage Agent** is an end-to-end machine learning system that not only classifies incoming emails into 5 distinct operational categories, but also couples the predictions to an **Agent / Action Layer** that executes appropriate predefined actions.

### Motivation Figures:
- 📧 **121** average emails received per day
- ⏱️ **28%** of the workday spent managing email
- 🤯 **62%** of workers feel overwhelmed by their inbox

---

## 2. Core Architecture & Pipeline

```text
Email Input (Sender, Subject, Body, Time, Links, Attachments)
    ↓
Text Preprocessing (Lowercasing, Regex, NLTK Tokenization, Stopwords, Stemming)
    ↓
Feature Extraction
    ├── Subject  → TF-IDF Vectorizer
    ├── Body     → Bag of Words (CountVectorizer + MaxAbsScaler)
    └── Metadata → Domain Encoding + Interaction History + Time + Links + Attachments
    ↓
ML Classifier
    ├── Naive Bayes (MultinomialNB)
    ├── Logistic Regression
    └── SVM with RBF Kernel
    ↓
Predicted Label (URGENT | NEEDS-REPLY | FYI | SPAM | DELEGATE)
    ↓
Agent / Action Layer (Decoupled Rule Engine)
    ├── URGENT      → Flag & Move to Priority Folder
    ├── NEEDS-REPLY → Draft Suggested Reply
    ├── FYI         → Auto-Archive Silently
    ├── SPAM        → Delete Automatically (Move to Trash)
    └── DELEGATE    → Classification Display (Automated Action Rule: To Be Defined)
    ↓
Result & Action Display (Interactive Streamlit Interface)
```

---

## 3. Email Categories & Agent Action Mapping

| Predicted Category | Agent Action Rule | Implementation & Simulated Execution |
| :--- | :--- | :--- |
| **`URGENT`** | Flag the email and move it to a priority folder | Moved to `Priority_Urgent` folder, starred as P0, priority notification alert triggered. |
| **`NEEDS-REPLY`** | Draft a suggested reply | Intelligent contextual reply drafted based on subject & sender, saved in `Drafts/`. |
| **`FYI`** | Auto-archive the email silently | Marked as read, routed to `Archive/` with notifications suppressed. |
| **`SPAM`** | Delete the email automatically | Moved to `Trash/` with 30-day purge schedule; sender domain blocked. |
| **`DELEGATE`** | Classify as an email requiring delegation handling | **Rule 9 & Section 6:** Clearly marked placeholder indicating the automated action rule is to be defined. Held in `Pending_Delegation` queue. |

*Disclaimer: In compliance with Project Rules 7 & 8, all operations are simulated safely in memory and audit logs.*

---

## 4. Directory Structure

```text
inbox-triage-agent/
│
├── app.py                      # Interactive Streamlit Web Interface
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
├── project.md                  # Academic report and specifications
│
├── data/
│   ├── raw/                    # Raw email dataset (emails_raw.csv)
│   ├── processed/              # Preprocessed dataset (emails_processed.csv)
│   └── generate_dataset.py     # Dataset synthesis and benchmarking script
│
├── models/
│   ├── feature_pipeline.joblib # Trained hybrid feature extraction pipeline
│   ├── naive_bayes/            # Trained Naive Bayes classifier
│   ├── logistic_regression/    # Trained Logistic Regression classifier
│   └── svm/                    # Trained SVM (RBF) classifier
│
├── src/
│   ├── preprocessing.py        # NLTK text cleaner and stemmer
│   ├── features.py             # TF-IDF, BoW, and metadata extractor
│   ├── train.py                # Model training and ablation pipeline
│   ├── predict.py              # Single and batch inference engine
│   ├── evaluation.py           # Multi-class evaluation metrics & confusion matrices
│   └── agent.py                # Decoupled agent action rule engine
│
├── tests/
│   └── test_triage.py          # Unit & integration tests
│
├── notebooks/
│   └── experimentation.ipynb   # Step-by-step EDA and experimentation notebook
│
└── outputs/
    ├── metrics/                # Benchmark JSON and CSV results
    └── figures/                # Confusion matrix heatmaps & ablation charts
```

---

## 5. Quickstart & Installation

### Step 1: Clone or Navigate to Directory
```bash
cd inbox-triage-agent
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Unit Tests
```bash
python -m unittest tests/test_triage.py
```

### Step 4: Launch Streamlit Application
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 6. Training and Evaluation

To regenerate the benchmark dataset and retrain all three core models:

```bash
# 1. Generate realistic benchmark dataset
python -m data.generate_dataset

# 2. Train Naive Bayes, Logistic Regression, SVM (RBF), and run Ablation Study
python -m src.train
```

### Empirical Model Performance:

| Model | Actual Accuracy | Macro F1 | Weighted F1 | PPT Reference | Difference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | **100.00%** | **100.00%** | **100.00%** | 78.0% | +22.00% |
| **Logistic Regression** | **100.00%** | **100.00%** | **100.00%** | 85.0% | +15.00% |
| **SVM (RBF Kernel)** | **100.00%** | **100.00%** | **100.00%** | 88.0% | +12.00% |

---

## 7. Feature Ablation Study

| Feature Modality | Feature Dimensions | Classification Accuracy |
| :--- | :---: | :---: |
| **Subject TF-IDF Only** | 500 | 100.00% |
| **Body Bag of Words Only** | 1,500 | 100.00% |
| **Text Only (Subject + Body)** | 2,000 | 100.00% |
| **Metadata Only (Domain, Time, Links, Attachments)** | 8 | 52.25% |
| **Full Hybrid Pipeline** | **2,008** | **100.00%** |

---

## 8. Definition of Done Checklist

- [x] Email data can be loaded or entered.
- [x] Email text is preprocessed using NLTK (lowercasing, stopword removal, stemming).
- [x] Subject and body features are generated (Subject: TF-IDF, Body: Bag of Words).
- [x] Required metadata features represented (Domain, History, Time, Links, Attachments).
- [x] Naive Bayes implemented and evaluated.
- [x] Logistic Regression implemented and evaluated.
- [x] SVM (RBF Kernel) implemented and evaluated.
- [x] Emails classified into all 5 categories (`URGENT`, `NEEDS-REPLY`, `FYI`, `SPAM`, `DELEGATE`).
- [x] Agent rules map classifications to decoupled actions with simulation disclosure.
- [x] Streamlit interface runs successfully with all specified sections.
- [x] Model evaluation metrics and confusion matrices displayed.
- [x] Model comparison with PPT reference targets available.
- [x] No unsupported results presented as factual.
- [x] Complete project documentation and notebook included.
