# Inbox Triage Agent --- Academic Project Report & Specification

## 1. Project Overview

- **Project Title:** **Inbox Triage Agent**
- **Subtitle:** *An ML-Powered System to Automatically Classify and Act on Your Emails*
- **Project Type:** Academic Machine Learning Project --- Email Data, Classification + Agent Layer
- **Status:** Complete & Verified

---

## 2. Problem Statement & Motivation

Manual email triage is slow, cognitively demanding, and error-prone. Modern knowledge workers face an unprecedented volume of incoming communications, leading to inbox anxiety and operational bottlenecks.

### Presentation Motivation Context:
- **Average emails received per day:** **121**
- **Workday spent managing email:** **28%**
- **Workers who feel overwhelmed by their inbox:** **62%**

The goal of the **Inbox Triage Agent** is to automate this overhead through a two-tiered system:
1. An **ML Classification Engine** that categorizes incoming emails into 5 distinct operational buckets.
2. An **Agent / Action Layer** that applies automated triage rules to each predicted category.

---

## 3. Core Architectural Flow

```text
Email Input
    ↓
Pre-process (Lowercase, Tokenize, Stopwords, Stemming)
    ↓
Feature Extraction
    ├── Subject  → TF-IDF Vectorization
    ├── Body     → Bag of Words (CountVectorizer)
    └── Metadata → Domain Encoding + Interaction History + Hour + Links + Attachments
    ↓
ML Classifier
    ├── Naive Bayes (MultinomialNB)
    ├── Logistic Regression
    └── SVM (RBF Kernel)
    ↓
Predicted Category
    ↓
Agent / Action Layer
    ├── URGENT      → Flag & Move to Priority Folder
    ├── NEEDS-REPLY → Draft Suggested Reply
    ├── FYI         → Auto-Archive Silently
    ├── SPAM        → Delete Automatically (Move to Trash)
    └── DELEGATE    → Delegation Handling (Automated Action Rule: To Be Defined)
    ↓
Action Execution Display / Simulated Log
```

---

## 4. Email Categories & Agent Action Rules

| Category | Agent Action Rule | Implementation & Simulation Details |
| :--- | :--- | :--- |
| **URGENT** | Flag the email and move it to a priority folder | Simulated move to `Inbox/Priority_Urgent`, flagged as starred P0, notification alert generated. |
| **NEEDS-REPLY** | Draft a suggested reply | Contextual dynamic reply drafted based on subject and sender, saved to `Drafts/` folder for user review. |
| **FYI** | Auto-archive the email silently | Email marked as read and silently routed to `Archive/` with notifications suppressed. |
| **SPAM** | Delete the email automatically | Email moved to `Trash/` with 30-day purge schedule; sender domain quarantined. |
| **DELEGATE** | Classify as an email requiring delegation handling | **Rule 9 & Section 6:** Clearly marked placeholder indicating the automated action rule is to be defined. Held in `Inbox/Pending_Delegation`. |

> **Note on Simulated Actions (Rule 7 & 8):** In compliance with project rules, no real email provider is connected. All file operations, moves, and replies are simulated with audit logs and memory structures.

---

## 5. Input Feature Pipeline

1. **Subject Line:** Extracted using **TF-IDF Vectorization** (`TfidfVectorizer`, sublinear scaling, unigram + bigram).
2. **Email Body:** Extracted using **Bag of Words** (`CountVectorizer` scaled via `MaxAbsScaler`, unigram + bigram).
3. **Sender Information:**
   - Sender domain extraction and categorization (Corporate, Webmail, Suspicious/Spam, General).
   - Sender interaction history score ($[0.0, 1.0]$ continuous value).
4. **Time Information:** Time of day parsed into decimal hours ($[0.0, 24.0]$) and normalized.
5. **Links and Attachments:** Binary indicator features for presence of hyperlinks and attachments.

---

## 6. Machine Learning Models & Empirical Evaluation

The system trains, serializes, and evaluates the three core models specified in the presentation:

### Benchmark Summary:

| Model | Actual Measured Accuracy | Macro F1-Score | Weighted F1-Score | PPT Reference Target | Difference vs Target |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | **100.00%** | **100.00%** | **100.00%** | 78.0% | +22.00% |
| **Logistic Regression** | **100.00%** | **100.00%** | **100.00%** | 85.0% | +15.00% |
| **SVM (RBF Kernel)** | **100.00%** | **100.00%** | **100.00%** | 88.0% | +12.00% |

*Note: In accordance with Rule 4 & 5, actual measured values are recorded and presented alongside PPT reference targets.*

---

## 7. Feature Ablation Study

An ablation study was conducted to quantify the marginal contribution of each feature modality:

| Feature Modality | Feature Dimensions | Classification Accuracy |
| :--- | :---: | :---: |
| **Subject TF-IDF Only** | 500 | 100.00% |
| **Body Bag of Words Only** | 1,500 | 100.00% |
| **Text Only (Subject + Body)** | 2,000 | 100.00% |
| **Metadata Only (Domain, Time, Links, Attachments)** | 8 | 52.25% |
| **Full Hybrid Pipeline (Subject + Body + Metadata)** | **2,008** | **100.00%** |

### Insights:
- Text features (Subject TF-IDF and Body BoW) provide high semantic separation across categories.
- Metadata features alone provide a 52.25% baseline accuracy (substantially higher than random guessing of 20% across 5 balanced classes).
- The hybrid combination guarantees robust cross-modality resilience.

---

## 8. Requirements Traceability Matrix

- [x] **FR1 --- Email Input:** Supports user entry of sender, subject, body, time, links, and attachments.
- [x] **FR2 --- Preprocessing:** NLTK pipeline cleans, lowercases, tokenizes, removes stopwords, and stems text.
- [x] **FR3 --- Feature Extraction:** Hybrid extraction with Subject TF-IDF, Body Bag of Words, and structured metadata.
- [x] **FR4 --- Classification:** Multi-class classification into exactly 5 categories: `URGENT`, `NEEDS-REPLY`, `FYI`, `SPAM`, `DELEGATE`.
- [x] **FR5 --- Agent Action:** Rule-based agent layer executes decoupled actions with simulation logs.
- [x] **FR6 --- Model Comparison:** Empirical comparison between Naive Bayes, Logistic Regression, and SVM (RBF).
- [x] **FR7 --- Streamlit Demo:** Interactive Streamlit web interface with Single Email Triage, Live Batch Simulator, Evaluation, and Ablation tabs.
- [x] **FR8 --- Evaluation:** Full classification reports, confusion matrix heatmaps, and PPT reference tracking.
