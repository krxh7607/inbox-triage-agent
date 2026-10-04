"""
Streamlit Web Application for Inbox Triage Agent.
An ML-Powered System to Automatically Classify and Act on Your Emails.

Academic Machine Learning Project --- Email Data, Classification + Agent Layer
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Inbox Triage Agent",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
        padding: 1.1rem;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: bold;
        color: #1E40AF;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #4B5563;
        font-weight: 500;
    }
    .badge-urgent {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
        border: 1px solid #F87171;
    }
    .badge-reply {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
        border: 1px solid #FCD34D;
    }
    .badge-fyi {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
        border: 1px solid #93C5FD;
    }
    .badge-spam {
        background-color: #F3F4F6;
        color: #374151;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
        border: 1px solid #9CA3AF;
    }
    .badge-delegate {
        background-color: #EDE9FE;
        color: #5B21B6;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
        border: 1px solid #C4B5FD;
    }
    .simulated-banner {
        background-color: #FFFBEB;
        border: 1px dashed #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        color: #B45309;
        font-size: 0.85rem;
        margin-top: 0.8rem;
    }
    .action-box {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Lazy load components
@st.cache_resource
def load_triage_system():
    from src.predict import EmailClassifierPredictor
    from src.agent import AgentActionEngine
    predictor = EmailClassifierPredictor(base_dir=BASE_DIR)
    agent = AgentActionEngine()
    return predictor, agent

@st.cache_data
def load_benchmark_metrics():
    metrics_path = os.path.join(OUTPUTS_DIR, "metrics", "model_comparison.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return []

@st.cache_data
def load_ablation_metrics():
    ablation_path = os.path.join(OUTPUTS_DIR, "metrics", "ablation_study.json")
    if os.path.exists(ablation_path):
        with open(ablation_path, "r") as f:
            return json.load(f)
    return []

@st.cache_data
def load_sample_dataset():
    raw_path = os.path.join(DATA_DIR, "raw", "emails_raw.csv")
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    return pd.DataFrame()

# Initialize resources
predictor, agent_engine = load_triage_system()
metrics_data = load_benchmark_metrics()
ablation_data = load_ablation_metrics()
df_samples = load_sample_dataset()

# Sidebar: Global Navigation & Settings
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/561/561127.png", width=64)
st.sidebar.markdown("### **Inbox Triage Agent**")
st.sidebar.caption("Machine Learning + Agent/Action Architecture")

navigation = st.sidebar.radio(
    "Navigation Menu",
    [
        "📨 Triage Single Email",
        "📥 Live Inbox Simulator (Batch)",
        "📊 Model Comparison & Evaluation",
        "🔬 Feature Ablation Study",
        "📖 System Architecture & Problem"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### **Active ML Model**")
active_model = st.sidebar.selectbox(
    "Select Classifier",
    ["SVM (RBF)", "Logistic Regression", "Naive Bayes"],
    index=0,
    help="Select which trained classifier powers the triage engine"
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### **Project Problem Context**")
st.sidebar.info(
    "**Motivation Figures:**\n"
    "- 📧 **121** emails/day average\n"
    "- ⏱️ **28%** of workday on triage\n"
    "- 🤯 **62%** workers overwhelmed"
)


# =========================================================================
# TAB 1: TRIAGE SINGLE EMAIL
# =========================================================================
if navigation == "📨 Triage Single Email":
    st.markdown('<div class="main-header">Inbox Triage Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An ML-Powered System to Automatically Classify and Act on Your Emails</div>', unsafe_allow_html=True)

    # Motivation context cards
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">121</div>
            <div class="metric-lbl">Average Emails Received / Day</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">28%</div>
            <div class="metric-lbl">Workday Spent Managing Email</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">62%</div>
            <div class="metric-lbl">Workers Feeling Overwhelmed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Email Presets for Demo
    presets = {
        "Custom Input": {
            "sender": "alex.mercer@company.com",
            "subject": "",
            "body": "",
            "time": "14:30",
            "links": False,
            "attach": False
        },
        "Example 1: URGENT - Server Database Outage": {
            "sender": "ops-alert@company.com",
            "subject": "CRITICAL: Primary database cluster outage in production",
            "body": "The PostgreSQL primary cluster is experiencing severe latency spikes and connection timeouts. Over 40% of checkout transactions are failing. All on-call engineers join emergency war room immediately.",
            "time": "14:15",
            "links": False,
            "attach": False
        },
        "Example 2: NEEDS-REPLY - Roadmap Feedback Sync": {
            "sender": "sarah.connor@company.com",
            "subject": "Review needed: proposal draft for Q4 product roadmap",
            "body": "Hi team, I attached the updated deck for Friday's client call. Could you please take a look and let me know your thoughts before tomorrow? Does Tuesday at 2 PM work for a sync?",
            "time": "10:30",
            "links": False,
            "attach": True
        },
        "Example 3: FYI - Weekly Engineering Digest": {
            "sender": "newsletter@company.com",
            "subject": "Weekly digest: platform release notes and scheduled maintenance",
            "body": "Here is the routine summary of team activities from this past week. Facilities management will be performing regular maintenance this Saturday. No action required on your part.",
            "time": "09:00",
            "links": True,
            "attach": False
        },
        "Example 4: SPAM - Free Lottery Prize Claim": {
            "sender": "claim-rewards@prizewinners.top",
            "subject": "Exclusive offer: Claim your free lottery cash prize voucher",
            "body": "Congratulations! Your email address was chosen to receive an exclusive cash grant of $50,000. Click here immediately to verify your profile details and prevent account closure.",
            "time": "02:45",
            "links": True,
            "attach": False
        },
        "Example 5: DELEGATE - Customer Billing Ticket": {
            "sender": "support-intake@company.com",
            "subject": "Customer ticket: Assign billing issue to specialist",
            "body": "We received an inbound customer support escalation regarding subscription billing overcharges. Can someone please route this request to the engineer on rotation?",
            "time": "11:20",
            "links": False,
            "attach": False
        }
    }

    selected_preset = st.selectbox(
        "Choose an Email Preset or Select 'Custom Input':",
        list(presets.keys()),
        index=1
    )

    preset_data = presets[selected_preset]

    col_input, col_meta = st.columns([2, 1])

    with col_input:
        st.markdown("#### **Email Information**")
        sender_val = st.text_input("Sender Email Address:", value=preset_data["sender"])
        subject_val = st.text_input("Subject Line (Processed with TF-IDF):", value=preset_data["subject"])
        body_val = st.text_area("Email Body (Processed with Bag of Words):", value=preset_data["body"], height=160)

    with col_meta:
        st.markdown("#### **Metadata Information**")
        time_val = st.text_input("Received Time (HH:MM):", value=preset_data["time"])
        links_val = st.checkbox("Contains Links", value=preset_data["links"])
        attach_val = st.checkbox("Contains Attachments", value=preset_data["attach"])

        st.markdown("##### **Model Selection**")
        model_choice = st.selectbox("Classifier to use:", ["SVM (RBF)", "Logistic Regression", "Naive Bayes"], index=0)

        triage_button = st.button("🚀 Run Inbox Triage", type="primary", use_container_width=True)

    # Classification & Action Execution
    if triage_button or subject_val.strip() != "":
        email_payload = {
            "email_id": f"EML-{datetime.now().strftime('%H%M%S')}",
            "sender": sender_val,
            "subject": subject_val,
            "body": body_val,
            "received_time": time_val,
            "has_links": 1 if links_val else 0,
            "has_attachments": 1 if attach_val else 0
        }

        with st.spinner(f"Running ML Classification with {model_choice}..."):
            result = predictor.predict_single(email_payload, model_name=model_choice, execute_agent=True)

        predicted_category = result["predicted_category"]
        confidence = result["confidence"]
        agent_action = result["agent_action"]
        probabilities = result.get("probabilities", {})

        st.markdown("---")
        st.markdown("### **Triage Execution Results**")

        res_col1, res_col2 = st.columns([1, 1.3])

        with res_col1:
            st.markdown("#### **Step 1: Machine Learning Classifier**")
            badge_class = {
                "URGENT": "badge-urgent",
                "NEEDS-REPLY": "badge-reply",
                "FYI": "badge-fyi",
                "SPAM": "badge-spam",
                "DELEGATE": "badge-delegate"
            }.get(predicted_category, "badge-fyi")

            st.markdown(f"**Predicted Category:** <span class='{badge_class}'>{predicted_category}</span>", unsafe_allow_html=True)
            st.markdown(f"**Model Employed:** `{model_choice}`")
            st.markdown(f"**Prediction Confidence:** `{confidence:.1%}`")

            # Probability Distribution Bar Chart
            if probabilities:
                st.markdown("##### **Class Probabilities Distribution**")
                prob_df = pd.DataFrame({
                    "Category": list(probabilities.keys()),
                    "Probability": list(probabilities.values())
                }).sort_values("Probability", ascending=True)

                fig, ax = plt.subplots(figsize=(6, 3.2))
                colors = ["#991B1B" if c == "URGENT" else "#92400E" if c == "NEEDS-REPLY" else "#1E40AF" if c == "FYI" else "#374151" if c == "SPAM" else "#5B21B6" for c in prob_df["Category"]]
                ax.barh(prob_df["Category"], prob_df["Probability"], color=colors, height=0.6)
                ax.set_xlim(0, 1.0)
                ax.set_xlabel("Predicted Probability")
                for i, v in enumerate(prob_df["Probability"]):
                    ax.text(v + 0.02, i, f"{v:.1%}", va="center", fontsize=9, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with res_col2:
            st.markdown("#### **Step 2: Agent / Action Layer**")
            st.markdown(f"**Automated Action Rule:** **{agent_action['action_description']}**")

            # Action Rendering depending on category
            sim = agent_action.get("simulated_execution", {})

            if predicted_category == "URGENT":
                st.error("🚨 **High Priority Action Taken:**")
                st.write(f"- 📁 Moved to Folder: `{sim.get('target_folder')}`")
                st.write(f"- ⭐ Flagged Status: `{sim.get('flagged')}`")
                st.write(f"- 🔔 Priority Escalation: `{sim.get('priority_level')}`")
                st.caption(agent_action["details"])

            elif predicted_category == "NEEDS-REPLY":
                st.warning("💬 **Reply Action Taken:**")
                st.write(f"- 📝 Draft ID: `{sim.get('draft_id')}`")
                st.write(f"- 📂 Target Folder: `{sim.get('target_folder')}`")
                st.markdown("**Suggested Reply Draft Generated by Agent:**")
                st.text_area(
                    "Review or Edit Draft Reply:",
                    value=sim.get("suggested_reply", ""),
                    height=130,
                    key="reply_draft_box"
                )
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.button("✉️ Approve & Send Draft (Simulated)", key="btn_send")
                with col_b2:
                    st.button("💾 Save as Draft", key="btn_save")

            elif predicted_category == "FYI":
                st.info("ℹ️ **Informational Action Taken:**")
                st.write(f"- 🗄️ Action: `{sim.get('status')}`")
                st.write(f"- 📁 Target Folder: `{sim.get('target_folder')}`")
                st.write(f"- 🔕 Notifications Suppressed: `{sim.get('notifications_suppressed')}`")
                st.caption(agent_action["details"])

            elif predicted_category == "SPAM":
                st.markdown("🗑️ **Spam Disposal Action Taken:**")
                st.write(f"- 🚮 Status: `{sim.get('status')}`")
                st.write(f"- 📁 Moved to: `{sim.get('target_folder')}`")
                st.write(f"- 🛡️ Quarantine: Sender `{sender_val}` blocked")
                st.caption(agent_action["details"])

            elif predicted_category == "DELEGATE":
                st.markdown("👥 **Delegation Handling Classification:**")
                st.write(f"- 🏷️ Status: `{sim.get('status')}`")
                st.write(f"- 📁 Held in Queue: `{sim.get('target_folder')}`")
                st.warning(
                    f"⚠️ **Specification Rule 9 Notice:**\n\n"
                    f"{sim.get('placeholder_notice')}"
                )

            # Simulated Disclaimer
            st.markdown("""
            <div class="simulated-banner">
                ℹ️ <strong>Simulated Action Execution:</strong> In compliance with Project Rule 7 & 8, no real email provider is connected. All file operations, folder moves, and replies are simulated in local application memory.
            </div>
            """, unsafe_allow_html=True)


# =========================================================================
# TAB 2: LIVE INBOX SIMULATOR (BATCH TRIAGE)
# =========================================================================
elif navigation == "📥 Live Inbox Simulator (Batch)":
    st.markdown('<div class="main-header">Live Inbox Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Demonstrate end-to-end triage on a batch of incoming unread emails</div>', unsafe_allow_html=True)

    if df_samples.empty:
        st.warning("No sample emails found. Please ensure `data/raw/emails_raw.csv` is generated.")
    else:
        # Take a slice of 15 emails for interactive triage
        batch_size = st.slider("Select batch size to triage:", min_value=5, max_value=25, value=12, step=1)
        df_batch = df_samples.head(batch_size).copy()

        col_act1, col_act2 = st.columns([2, 1])
        with col_act1:
            st.markdown(f"**Loaded {len(df_batch)} incoming unorganized emails from the inbox queue.**")
        with col_act2:
            run_batch_btn = st.button("⚡ Triage Entire Inbox with Agent", type="primary", use_container_width=True)

        if run_batch_btn:
            with st.spinner("Triaging emails and executing agent action rules..."):
                triaged_df = predictor.predict_batch(df_batch, model_name=active_model, execute_agent=True)

            # Statistics summary
            c_counts = triaged_df["predicted_category"].value_counts().to_dict()

            col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
            col_s1.metric("🚨 URGENT", c_counts.get("URGENT", 0), "Flagged")
            col_s2.metric("💬 NEEDS-REPLY", c_counts.get("NEEDS-REPLY", 0), "Drafts Created")
            col_s3.metric("ℹ️ FYI", c_counts.get("FYI", 0), "Archived")
            col_s4.metric("🗑️ SPAM", c_counts.get("SPAM", 0), "Deleted")
            col_s5.metric("👥 DELEGATE", c_counts.get("DELEGATE", 0), "Pending Rule")

            st.markdown("---")
            st.markdown("#### **Triaged Inbox Overview**")

            category_filter = st.multiselect(
                "Filter by Category:",
                options=["URGENT", "NEEDS-REPLY", "FYI", "SPAM", "DELEGATE"],
                default=["URGENT", "NEEDS-REPLY", "FYI", "SPAM", "DELEGATE"]
            )

            filtered_df = triaged_df[triaged_df["predicted_category"].isin(category_filter)]

            display_cols = ["email_id", "sender", "subject", "received_time", "predicted_category", "confidence", "agent_action"]
            styled_df = filtered_df[display_cols].copy()
            styled_df["confidence"] = styled_df["confidence"].apply(lambda x: f"{x:.1%}")

            st.dataframe(styled_df, use_container_width=True)

            st.markdown("""
            <div class="simulated-banner">
                ℹ️ All bulk triage actions above are <strong>simulated actions</strong> executed by the agent layer.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("#### **Current Untriaged Inbox Queue**")
            st.dataframe(df_batch[["email_id", "sender", "subject", "received_time", "has_links", "has_attachments"]], use_container_width=True)


# =========================================================================
# TAB 3: MODEL COMPARISON & EVALUATION
# =========================================================================
elif navigation == "📊 Model Comparison & Evaluation":
    st.markdown('<div class="main-header">Model Comparison & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical classification metrics comparing Naive Bayes, Logistic Regression, and SVM (RBF)</div>', unsafe_allow_html=True)

    if not metrics_data:
        st.warning("No evaluation metrics found. Please run `python -m src.train` first.")
    else:
        # Comparison Table with Presentation Targets
        st.markdown("### **1. Summary Performance vs. Presentation Reference Targets**")
        st.caption("Reference targets from initial presentation: Naive Bayes: 78% | Logistic Regression: 85% | SVM (RBF): 88%")

        table_rows = []
        for m in metrics_data:
            ref_str = f"{m['presentation_reference'] * 100:.1f}%" if m['presentation_reference'] else "N/A"
            diff_str = f"{m['diff_from_reference'] * 100:+.2f}%" if m['diff_from_reference'] is not None else "N/A"
            table_rows.append({
                "Model": m["model_name"],
                "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Macro F1": f"{m['f1_macro'] * 100:.2f}%",
                "Weighted F1": f"{m['f1_weighted'] * 100:.2f}%",
                "Macro Precision": f"{m['precision_macro'] * 100:.2f}%",
                "Macro Recall": f"{m['recall_macro'] * 100:.2f}%",
                "PPT Reference": ref_str,
                "Difference": diff_str
            })

        df_summary = pd.DataFrame(table_rows)
        st.table(df_summary)

        # Bar chart comparing Actual Accuracy vs PPT Reference Target
        col_c1, col_c2 = st.columns([1.2, 1])
        with col_c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            model_names = [r["model_name"] for r in metrics_data]
            actual_accs = [r["accuracy"] * 100 for r in metrics_data]
            ref_accs = [r["presentation_reference"] * 100 for r in metrics_data]

            x = np.arange(len(model_names))
            width = 0.35

            rects1 = ax.bar(x - width/2, actual_accs, width, label="Actual Achieved Accuracy", color="#2563EB")
            rects2 = ax.bar(x + width/2, ref_accs, width, label="Presentation Target", color="#DC2626", alpha=0.8)

            ax.set_ylabel("Accuracy (%)", fontweight="bold")
            ax.set_title("Model Accuracy: Actual vs. Reference Benchmark", fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, fontweight="bold")
            ax.set_ylim(0, 110)
            ax.legend(loc="lower right")

            for rect in rects1:
                h = rect.get_height()
                ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
            for rect in rects2:
                h = rect.get_height()
                ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_c2:
            st.markdown("#### **Evaluation Key Observations**")
            st.markdown("""
            - **SVM (RBF Kernel)** achieves robust generalization on the non-linear feature boundary formed by combined sparse text and dense metadata.
            - **Logistic Regression** provides high linear discriminability with efficient inference times.
            - **Naive Bayes** serves as a fast baseline model using multinomial count distributions.
            - All models conform to the project specification requirements and are saved in `models/` directory for reproducibility.
            """)

        st.markdown("---")
        st.markdown("### **2. Confusion Matrix Heatmaps**")

        c_tabs = st.tabs([m["model_name"] for m in metrics_data])
        for idx, tab in enumerate(c_tabs):
            with tab:
                m_info = metrics_data[idx]
                cm = np.array(m_info["confusion_matrix"])
                labels = m_info["labels"]

                cm_col1, cm_col2 = st.columns([1, 1.2])
                with cm_col1:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
                    ax.set_title(f"Confusion Matrix - {m_info['model_name']}", fontweight="bold")
                    ax.set_xlabel("Predicted Label", fontweight="bold")
                    ax.set_ylabel("True Label", fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with cm_col2:
                    st.markdown(f"#### **Per-Class Classification Report: {m_info['model_name']}**")
                    report = m_info["classification_report"]
                    report_rows = []
                    for lbl in labels:
                        if lbl in report:
                            report_rows.append({
                                "Category": lbl,
                                "Precision": f"{report[lbl]['precision'] * 100:.1f}%",
                                "Recall": f"{report[lbl]['recall'] * 100:.1f}%",
                                "F1-Score": f"{report[lbl]['f1-score'] * 100:.1f}%",
                                "Test Support": int(report[lbl]["support"])
                            })
                    st.dataframe(pd.DataFrame(report_rows), use_container_width=True)


# =========================================================================
# TAB 4: FEATURE ABLATION STUDY
# =========================================================================
elif navigation == "🔬 Feature Ablation Study":
    st.markdown('<div class="main-header">Feature Ablation Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical study analyzing the marginal contribution of each feature modality</div>', unsafe_allow_html=True)

    st.markdown("""
    The system utilizes a hybrid feature pipeline comprising:
    1. **Subject Line** extracted via **TF-IDF**
    2. **Email Body** extracted via **Bag of Words (CountVectorizer)**
    3. **Metadata Features** (Domain categorization, Sender interaction history, Hour of day, Has links, Has attachments)
    """)

    if not ablation_data:
        st.warning("No ablation study data found. Please run `python -m src.train` first.")
    else:
        df_abl = pd.DataFrame(ablation_data)

        col_a1, col_a2 = st.columns([1.2, 1])

        with col_a1:
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            y_pos = np.arange(len(df_abl))
            accs = df_abl["Accuracy"] * 100
            colors = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#8B5CF6"]

            bars = ax.barh(y_pos, accs, color=colors, height=0.55)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_abl["Feature Subset"], fontweight="bold")
            ax.set_xlabel("Classification Accuracy (%)", fontweight="bold")
            ax.set_xlim(0, 110)
            ax.set_title("Ablation Study: Impact of Feature Subsets", fontweight="bold")

            for bar in bars:
                w = bar.get_width()
                ax.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_a2:
            st.markdown("#### **Ablation Data Summary**")
            table_abl = df_abl[["Feature Subset", "Features Count", "Accuracy"]].copy()
            table_abl["Accuracy"] = table_abl["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
            st.table(table_abl)

        st.markdown("---")
        st.markdown("#### **Ablation Insights & Technical Rationale**")
        st.markdown("""
        - **Subject TF-IDF:** High informational density; concise subject keywords provide strong signals for urgent incident alerts and newsletter digests.
        - **Body Bag of Words:** Rich vocabulary representation capturing conversational nuance and specific request phrasing.
        - **Structured Metadata:** Metadata alone provides strong baseline discriminability (e.g. spam domains and off-hours timestamps correlate heavily with unsolicited mail).
        - **Full Hybrid Pipeline:** Combining Subject TF-IDF, Body Bag of Words, and structured metadata maximizes robustness against adversarial or edge-case emails.
        """)


# =========================================================================
# TAB 5: SYSTEM ARCHITECTURE & PROBLEM
# =========================================================================
elif navigation == "📖 System Architecture & Problem":
    st.markdown('<div class="main-header">System Architecture & Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inbox Triage Agent --- Academic Project Documentation</div>', unsafe_allow_html=True)

    st.markdown("### **1. Main Pipeline Flow**")
    st.markdown("""
    ```text
    Email Input
        ↓
    Pre-process (NLTK Tokenization, Stopwords Removal, Stemming)
        ↓
    Feature Extraction (Subject TF-IDF + Body BoW + Metadata Scaled)
        ↓
    ML Classify (Naive Bayes | Logistic Regression | SVM RBF)
        ↓
    Agent / Action Layer (Decoupled Rule Mapping)
        ↓
    Action on Email (Priority Flag | Reply Draft | Auto-Archive | Delete | Delegate Placeholder)
    ```
    """)

    st.markdown("---")
    st.markdown("### **2. Predefined Category to Action Rules (Section 6)**")

    st.markdown("""
    | Classification | Agent Action Rule | Status in Application |
    | :--- | :--- | :--- |
    | **URGENT** | Flag the email and move it to a priority folder | Implemented (Simulated priority folder move & starred status) |
    | **NEEDS-REPLY** | Draft a suggested reply | Implemented (Simulated contextual reply generation in Drafts) |
    | **FYI** | Auto-archive the email silently | Implemented (Simulated silent archival without notification) |
    | **SPAM** | Delete the email automatically | Implemented (Simulated move to Trash & quarantine) |
    | **DELEGATE** | Classify as an email requiring delegation handling | Implemented (Clearly marked placeholder; automated action to be defined per Rule 9) |
    """)

    st.markdown("---")
    st.markdown("### **3. Dataset Dynamics (Section 11)**")
    st.markdown("""
    The system is trained and benchmarked on emails capturing patterns from:
    - **Enron Email Corpus:** Realistic organizational communication dynamics, executive requests, project hand-offs, and operational scheduling.
    - **SpamAssassin Benchmark:** Realistic spam patterns, phishing lures, and domain anomalies.
    - **Personal & Corporate Triage Flows:** Standard administrative newsletters, bug escalations, and meeting requests.
    """)

    st.markdown("---")
    st.markdown("### **4. Technology Stack & Verification**")
    st.markdown("""
    - **Language:** Python 3.13
    - **Machine Learning:** scikit-learn (TF-IDF, CountVectorizer, Naive Bayes, Logistic Regression, SVM RBF, Evaluators)
    - **NLP:** NLTK (Tokenization, Porter Stemmer, Stopwords)
    - **User Interface:** Streamlit
    - **Data Handling:** pandas, numpy, scipy
    - **Visualization:** matplotlib, seaborn
    """)
