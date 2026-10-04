# Deployment Guide --- Inbox Triage Agent

This guide walks you through deploying your **Inbox Triage Agent** to the web so anyone can access it via a public URL.

---

## 🌟 Method 1: Streamlit Community Cloud (Recommended --- 100% Free & Fastest)

Streamlit Community Cloud is the official and simplest way to deploy Streamlit apps with zero server configuration.

### Steps:
1. **Push your code to GitHub:**
   ```bash
   cd inbox-triage-agent
   git init
   git add .
   git commit -m "Initial commit of Inbox Triage Agent"
   # Create a repository on GitHub, then link and push:
   git branch -M main
   git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Community Cloud:**
   - Go to **[share.streamlit.io](https://share.streamlit.io/)** and log in with your GitHub account.
   - Click **"New app"**.
   - Select your repository (`<YOUR-REPO-NAME>`), branch (`main`), and set the main file path to:
     ```text
     app.py
     ```
   - Click **"Deploy!"**.
   - Your app will be live with a permanent URL like `https://<your-app-name>.streamlit.app` in under 2 minutes!

---

## 🤗 Method 2: Hugging Face Spaces (Free & Popular for ML Projects)

Hugging Face Spaces provides free hosting with native Streamlit support.

### Steps:
1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)** and click **"Create new Space"**.
2. Space SDK: Select **Streamlit**.
3. Space hardware: Select **Free CPU basic**.
4. Clone the space repo locally or push your files:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR-USERNAME>/<YOUR-SPACE-NAME>
   git push space main
   ```
5. Hugging Face will automatically install `requirements.txt` and launch `app.py`.

---

## 🐳 Method 3: Docker Container Deployment (Render / Railway / Fly.io / Cloud)

A production-ready `Dockerfile` and `.dockerignore` are pre-configured in this repository.

### Option A: Local or Server Docker Run
```bash
# Build the Docker image
docker build -t inbox-triage-agent .

# Run the container mapping port 8501
docker run -p 8501:8501 inbox-triage-agent
```
Open `http://localhost:8501` in your browser.

### Option B: Deploy to Render.com
1. Create a free account on **[render.com](https://render.com/)**.
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Environment: Select **Docker** (Render will automatically detect the `Dockerfile`).
5. Plan: Select **Free**.
6. Click **Create Web Service**. Your app will be deployed with a free `onrender.com` URL.

### Option C: Deploy to Railway.app
1. Go to **[railway.app](https://railway.app/)**.
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Railway automatically detects the `Dockerfile` and deploys it on an assigned public domain.

---

## 🔒 Pre-deployment Verification Checklist

- [x] Pre-trained models serialized in `models/` (`feature_pipeline.joblib`, `naive_bayes/`, `logistic_regression/`, `svm/`)
- [x] Clean requirements pinned in `requirements.txt`
- [x] Auto-downloading of NLTK corpora (`stopwords`, `punkt`, `punkt_tab`) configured
- [x] Headless mode and port configuration set in `.streamlit/config.toml`
- [x] Unit tests passing (`python -m unittest tests/test_triage.py`)
