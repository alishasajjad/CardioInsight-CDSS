# CardioInsight CDSS

**AI-Powered Clinical Decision Support System for Heart Disease Risk Assessment**

---

## Project Overview

CardioInsight is a clinical decision support **demonstration** that combines ensemble machine learning, explainable AI, rule-based clinical recommendations, PDF medical reporting, secure user history, and a retrieval-augmented (RAG) health assistant. The system is trained exclusively on four clinically aligned angiographic datasets and is designed for academic presentation, portfolio showcase, and educational use.

> **Disclaimer:** This application is for educational purposes only. It is not a medical device and does not provide diagnosis or treatment.

> 📘 **New to the code?** See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture.

---

## Key Features

- **Ensemble ML prediction** — Random Forest, XGBoost, and ANN with ROC-AUC–weighted averaging
- **SHAP explainability** — Per-patient feature attribution
- **Clinical recommendation engine** — Low / medium / high risk action plans
- **PDF medical reports** — Hospital-style downloadable assessments
- **User authentication** — Secure registration, login, and session management
- **Prediction history** — Filterable record of past assessments
- **RAG health assistant** — Groq LLM grounded in a FAISS medical knowledge base
- **Modern healthcare UI** — Professional Streamlit dashboard (Refined Clinical Blue theme)

---

## System Architecture

The codebase is split into three independent areas that run in **one Streamlit process** (no separate API server): the frontend imports the backend's functions directly, and the ML layer publishes a **copy** of the trained model into the backend.

```
machine_learning/  ──(publishes model copy)──▶  backend/models/
                                                     │ imports + shared `system` dict
Streamlit frontend/  ──▶  Ensemble ML + SHAP ──▶ Recommendations + PDF
                     ──▶  SQLite (users, predictions, chat)
                     ──▶  FAISS + Groq (RAG assistant)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full diagram and data-flow.

---

## Technology Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.10+ |
| ML | scikit-learn, XGBoost, SHAP |
| UI | Streamlit |
| Database | SQLite |
| PDF | ReportLab |
| LLM | Groq API (Llama 3.x) |
| RAG | FAISS, sentence-transformers |
| Auth | bcrypt |

Dependencies are **pinned** to tested versions in `requirements.txt` (an unpinned scikit-learn upgrade previously broke model unpickling / SHAP — pinning prevents that drift).

---

## Machine Learning Models

| Model | Role |
|-------|------|
| Random Forest | Ensemble member; robust tabular baseline |
| XGBoost | Ensemble member; gradient boosting |
| ANN (MLP) | Ensemble member; non-linear patterns |

All models are tuned with `RandomizedSearchCV` and 5-fold stratified cross-validation.

**Deployment selection:** the model with the best hold-out ROC-AUC is deployed, with XGBoost preferred when it is within 0.005 ROC-AUC of the best. The chosen model is recorded in `backend/models/deployment/model_metadata.json` and can change on retrain. *(Currently deployed: **Random Forest**.)*

---

## Explainable AI (SHAP)

Tree SHAP values are computed on the deployed tree model (currently **Random Forest**) for each assessment, highlighting which clinical features increase or decrease the predicted risk for that specific patient.

---

## Ensemble Prediction System

Individual model probabilities are combined using ROC-AUC–normalized weights:

```
ensemble_prob = Σ (weight_i × prob_i) / Σ weight_i
```

The dashboard displays per-model probabilities, weights, agreement score, and an ensemble rationale. At inference the input feature columns are validated against the trained model contract (`model_metadata.json`) to prevent silent train/serve drift.

---

## Clinical Recommendation Engine

| Risk tier | Probability | Guidance |
|-----------|-------------|----------|
| Low | < 35% | Preventive lifestyle advice |
| Medium | 35–55% | Enhanced monitoring |
| High | > 55% | Cardiologist referral, ECG, stress test, lipid profile, BP evaluation |

---

## RAG-Powered AI Health Assistant

- Documents ingested from `backend/knowledge_base/` (AHA-style, WHO, clinical references)
- Chunked (≈500 chars, 80 overlap), embedded with `all-MiniLM-L6-v2`, and indexed in FAISS
- Groq LLM generates answers grounded in the retrieved medical content + the patient's prediction context
- Models: `llama-3.3-70b-versatile` (default), `llama-3.1-8b-instant`

---

## Authentication & Prediction History

- bcrypt password hashing (passwords are never stored in plain text)
- Session tokens in SQLite with **7-day expiry**
- Predictions, PDF paths, and chat messages stored per user (isolated by `user_id`)
- History page supports date filtering and session reload

---

## PDF Medical Reports

Each assessment generates a ReportLab PDF containing patient inputs, ensemble results, SHAP factors, recommendations, AI narrative, timestamp, and medical disclaimer.

---

## Dataset Information

### Included (clinical alignment)

| Dataset | Description |
|---------|-------------|
| heart_cleveland_upload | Cleveland heart disease records |
| Heart_disease_statlog | Statlog heart dataset |
| heart_disease_uci | UCI multi-site records |
| heart_statlog_cleveland_hungary_final | Cleveland/Hungary statlog |

### Excluded

Framingham, cardio_train, and redundant heart.csv — non-aligned labels or schema.

**Processed output:** `machine_learning/datasets/processed/clinical_heart_disease.csv` (~1,001 records after deduplication; 13 clinical + 4 engineered = 17 features).

---

## Folder Structure

Three independent areas — **machine learning**, **backend** (services), and **frontend** (UI). The trained model is *published as a copy* into the backend; the app never reads models out of the ML folder.

```
CardioInsight-CDSS/
├── machine_learning/          # ML work: data, training, model artifacts
│   ├── datasets/raw/          # Source CSV files (4 clinical)
│   ├── datasets/processed/    # Unified clinical dataset
│   ├── pipeline/              # loaders, preprocessing, eda, training, report
│   ├── models/                # Trained artifacts (source of truth)
│   ├── outputs/               # EDA & training figures + metrics
│   ├── config.py              # ML paths + feature/training constants
│   └── run_pipeline.py        # Train, deploy, publish model → backend
├── backend/                   # Services (the app's logic) — one folder per module
│   ├── auth/                  # bcrypt auth + sessions
│   ├── database/              # SQLite layer
│   ├── ensemble/              # Weighted ensemble inference
│   ├── llm/                   # Groq LLM client
│   ├── rag/                   # FAISS retrieval + RAG
│   ├── recommendations/       # Clinical rule engine
│   ├── reporting/             # ReportLab PDF
│   ├── models/                # COPY of deployed bundle (app reads this)
│   ├── assets/figures/        # COPY of charts for Analytics
│   ├── knowledge_base/        # RAG source documents
│   ├── scripts/               # DB migration, KB rebuild
│   └── config.py              # Runtime paths + feature/threshold contract
├── frontend/                  # Streamlit UI
│   ├── streamlit_app.py       # Entry point
│   ├── views/                 # Page modules + prediction_utils
│   ├── ui/                    # theme, components, session
│   └── .streamlit/config.toml # Theme/settings when launched from frontend/
├── data/                      # Runtime: SQLite DB, PDFs, FAISS index (gitignored)
├── docs/                      # Architecture, VIVA guide, training summary
├── .streamlit/config.toml     # Pinned light theme + settings (project-root launch)
└── requirements.txt
```

---

## Installation Guide

```bash
git clone <repository-url>
cd CardioInsight-CDSS
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Configuration Instructions

1. Copy `.env.example` to `.env` and add your Groq API key (the `.env` file is gitignored):
   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```
   You can also paste the key into the app's **Health Assistant → settings** box (kept per-session only).
2. Obtain a key at [console.groq.com](https://console.groq.com/)

> Prediction, SHAP, and PDF reports work **without** a key — only the AI assistant needs it.

---

## Running the Application

```bash
# 1. Train models, deploy, and publish a copy into the backend
python machine_learning/run_pipeline.py

# 2. Launch dashboard
streamlit run frontend/streamlit_app.py
```

> Launch from the **project root** (or from `frontend/` — both contain a `.streamlit/config.toml`) so the light theme loads. The pre-trained models are already published in `backend/models/`, so step 1 is only needed to (re)train.

Optional utilities:

```bash
python backend/scripts/migrate_db.py                    # initialize/repair the SQLite schema
python backend/scripts/build_knowledge_base.py --force  # rebuild the FAISS RAG index
```

---

## Building the Knowledge Base

1. Add `.md`, `.txt`, or `.json` files to `backend/knowledge_base/`
2. Rebuild the index:
   ```bash
   python backend/scripts/build_knowledge_base.py --force
   ```
   Or use **Admin → Rebuild vector index** in the app.

---

## Usage Workflow

1. **Register / Sign in**
2. **Risk Assessment** — enter clinical profile → run ensemble analysis
3. Review results, SHAP chart, recommendations
4. **Download PDF report**
5. **Health Assistant** — ask follow-up questions (RAG-grounded)
6. **History** — review past assessments

---

## Model Performance Results

Hold-out test set (clinical cohort, ~20% split):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| **Random Forest** | 0.841 | 0.891 | 0.811 | 0.849 | **0.910** |
| XGBoost | 0.841 | 0.891 | 0.811 | 0.849 | 0.903 |
| ANN | 0.831 | 0.847 | 0.847 | 0.847 | 0.907 |

**Deployed model:** Random Forest (best hold-out ROC-AUC). XGBoost is preferred only when within 0.005 of the best. Exact numbers may shift slightly on retrain.

---

## Future Enhancements

- Probability calibration (Platt scaling)
- Admin roles + login rate-limiting; encryption-at-rest for PHI use
- Automated tests / CI and Docker deployment
- FHIR / EHR integration
- Prospective clinical validation

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not found | Run `python machine_learning/run_pipeline.py` |
| Groq errors | Verify `GROQ_API_KEY` in `.env` (or enter it in the assistant settings) |
| Empty RAG answers | Run `python backend/scripts/build_knowledge_base.py --force` |
| Database errors | Run `python backend/scripts/migrate_db.py` |
| `torchvision` tracebacks in console | Harmless; suppressed by `fileWatcherType="none"` in `.streamlit/config.toml` |
| Dark inputs / wrong theme | Launch from a folder with `.streamlit/config.toml` (project root or `frontend/`), then hard-refresh the browser |
| Slow first load | Model + embedding-model download/caching — normal on first startup |

---

## Disclaimer

**CardioInsight CDSS is an educational clinical decision support demonstration.** It does not provide medical diagnosis, treatment, prescription, or emergency care. All outputs must be reviewed by qualified healthcare professionals. In a medical emergency, contact local emergency services immediately.

## Demo Video
- https://youtu.be/67vw09mdZUA
