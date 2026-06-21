# CardioInsight CDSS — Poora Project Samjho (Viva Guide)

> Yeh guide Roman Urdu + English mein hai. Har module, har important function, aur
> RAG / rebuild / ensemble / SHAP jaisi cheezein simple lafzon mein samjhayi gayi hain.
> Viva ke liye: pehle "Ek Line Mein" parho, phir detail, aur aakhir mein **Viva Q&A**.

---

## 0. Ek Line Mein Project

**CardioInsight CDSS** ek *educational* heart-disease risk prediction system hai. User apni
clinical details (age, BP, cholesterol, etc.) daalta hai → 3 machine learning models milkar
(ensemble) risk % batate hain → **SHAP** se samjhaya jaata hai kis cheez ne risk barhaya/ghataya
→ rule-based **recommendations** → ek **PDF report** → aur ek **AI chatbot (RAG + Groq)** jo
medical knowledge se grounded jawab deta hai. Sab kuch **SQLite** mein save hota hai.

> ⚠️ Yeh medical device NAHI hai — sirf educational/portfolio project hai.

---

## 1. Project ka Structure (3 hisson mein)

```
CardioInsight-CDSS/
├── machine_learning/   → ML ka kaam: data, training, models banana
├── backend/            → "dimagh": services (auth, db, prediction, rag, pdf)
└── frontend/           → "shakal": Streamlit UI (jo screen pe dikhta hai)
```

**Inko jodta kaun hai?** (Viva ka favourite sawaal)
- Yeh **ek hi Python process** hai (Streamlit app) — koi alag server/API nahi.
- `frontend/` apne andar `backend/` ko **import** karta hai aur uske functions seedha call karta hai.
- Connection ka asal point: `frontend/streamlit_app.py` mein `sys.path` me project root add hota hai,
  phir `load_system()` (cached) backend ke models + DB load karke ek `system` dict banata hai jo
  har page ko diya jaata hai: `PAGE_RENDERERS[nav](system)`.
- **machine_learning → backend**: ML training ke baad model ki **copy** `backend/models/` mein
  bhej deta hai (publish). Backend kabhi ML folder se model nahi parhta — apni local copy use karta hai.

```
machine_learning/  ──(model ki copy bhejta hai)──▶  backend/models/
                                                          │ imports + system dict
                                                     frontend/ (Streamlit UI)
```

---

## 2. Technology Stack (kaunsi cheez kis liye)

| Cheez | Kaam |
|------|------|
| **Streamlit** | Web UI banane ke liye (Python se hi front-end) |
| **scikit-learn** | Random Forest, ANN (MLP), tuning, metrics |
| **XGBoost** | Gradient boosting model |
| **SHAP** | Explainability — kis feature ne kitna asar dala |
| **SQLite** | Database (users, predictions, chat) |
| **bcrypt** | Password ko securely hash karna |
| **ReportLab** | PDF report banana |
| **Groq API** | LLM (Llama 3.x) — AI chatbot ke jawab |
| **sentence-transformers** | Text ko numbers (embeddings) mein badalna |
| **FAISS** | Embeddings mein se similar text dhoondhna (fast search) |

---

## 3. Data Pipeline — `machine_learning/`

Yeh poora flow ek command se chalta hai: **`python machine_learning/run_pipeline.py`**

### 3.1 `pipeline/loaders.py` — raw data load karna
4 alag-alag heart datasets (Cleveland, Statlog, UCI, Statlog-Cleveland-Hungary) ke column names
alag hain. Loaders unko ek standard shape mein laate hain.
- `load_heart_csv(path, target_col)` → Cleveland/Statlog CSV parhta hai, columns rename karta hai.
- `load_heart_disease_uci(path)` → UCI dataset (string values ko numbers mein map karta hai).
- `load_heart_statlog_hungary(path)` → 4th dataset.
- `clip_outliers(df)` → ghalat/extreme values ko physiological range mein clip karta hai
  (e.g. age 1–120, BP 50–250).
- `remove_duplicates(df)` → repeat rows hatata hai.
- Target ko **binary** banata hai: `(disease > 0) → 1`, warna `0`.

### 3.2 `pipeline/preprocessing.py` — saaf data taiyar karna
- `build_clinical_dataset()` → 4 dataset load → jodta hai → clip → duplicate hatata →
  features banata → missing values bhar deta → ek final CSV save karta hai.
- **Result:** ~**1001 rows**, ~**55% disease** (raw ~2677 they; dedup ke baad 1001).
- `engineer_clinical_features(df)` → **4 naye (engineered) features** banata hai:
  - `chol_bp_ratio` = cholesterol / resting_bp (lipid-pressure ratio)
  - `age_group` = age ko 4 groups mein (0–40, 40–55, 55–65, 65+)
  - `hr_reserve` = 220 − age − max_heart_rate (heart ka reserve)
  - `vessel_thal_score` = major_vessels + thalassemia
- **Missing values**: `SimpleImputer(strategy="median")` — har column ka median bhar deta hai.

> **Total features = 17** = 13 clinical + 4 engineered. (Yeh `FEATURE_COLUMNS` hai — bohat important.)

### 3.3 `pipeline/eda.py` — data ko samajhna (graphs)
- `run_eda()` → graphs banata hai (class balance, distributions, correlation heatmap) aur
  ek `eda_insights.json` save karta hai (mean age, disease rate, etc.).

### 3.4 `pipeline/training.py` — models train karna (dil ka kaam)
- `tune_random_forest / tune_xgboost / tune_ann` → har model ke liye **RandomizedSearchCV +
  5-fold cross-validation** se best settings dhoondhta hai (scoring = ROC-AUC).
- `train_and_deploy()` → poora training karta hai:
  1. Data ko **80/20** train/test mein baant-ta hai (stratified).
  2. 3 models train: **Random Forest, XGBoost, ANN (neural network/MLP)**.
  3. Test set pe **metrics** nikaalta hai: accuracy, precision, recall, F1, **ROC-AUC**.
  4. **Deployment model choose**: jo best ROC-AUC de — lekin agar XGBoost best se 0.005 ke andar
     ho to XGBoost ko prefer karta hai. (Abhi retrain ke baad **Random Forest** deploy hua,
     ROC-AUC ≈ **0.910**.)
  5. **SHAP** values + feature importance compute karke save karta hai.
  6. `model_metadata.json` save karta hai (metrics, ensemble weights, deployed model ka naam).
- `compute_shap(...)` + `_shap_positive_class_2d(...)` → SHAP values ko sahi shape mein laata hai
  (naye SHAP/XGBoost 3-D array dete hain — yeh function usko handle karta hai).

### 3.5 `run_pipeline.py` — sab ko chalana + publish
1. dataset banata → EDA → train → summary likhta.
2. **Publish**: trained model bundle + charts ki **copy** `backend/models/` aur
   `backend/assets/figures/` mein bhej deta hai.
3. DB initialize + RAG index (best-effort) bana deta hai.

---

## 4. Models aur Prediction kaise hoti hai — `backend/ensemble/`

### Ensemble kya hai?
3 models ka **milay-jhulay** (combined) faisla. Akele se zyada reliable. Har model risk %
deta hai, phir unka **weighted average** liya jaata hai.

### `ensemble.py` ke functions
- `load_all_models()` → RF, XGBoost, ANN + metadata + artifacts load karta hai (backend/models se).
- `compute_ensemble_weights(metadata)` → weights nikaalta hai. Yeh **ROC-AUC ke hisaab se
  normalize** hote hain (jo model behtar, uska weight thoda zyada). Agar metadata na ho to
  config ke default weights.
- `predict_single(model, X)` → ek model ka `predict_proba` → risk probability.
- `ensemble_predict(X, system)` → **asal prediction function**:
  - Pehle **safety check**: input ke columns model ke trained `feature_columns` se match karne
    chahiye (warna error — taake galat prediction na ho).
  - Har model se probability leta hai, weighted average karta hai → `ensemble_probability`.
  - `confidence` = models kitne aapas mein agree karte hain (agreement zyada → confidence zyada).
  - Agar koi model load na ho to **error** deta hai (fake 50% nahi deta).

> **Decision boundary**: probability ≥ 0.5 → "disease indicated". Lekin **risk tiers** alag hain
> (neeche section 6).

---

## 5. SHAP — Explainability (Viva ka important topic)

**SHAP kya hai?** Yeh batata hai ke **kis feature ne is khaas patient ka risk kitna barhaya ya
ghataya**. Har feature ko ek number milta hai:
- **+ (red)** → risk barhaya
- **− (green)** → risk ghataya
- `base value` (≈0.5) = model ka average output; uske upar features ne push kiya.

`shap_for_instance(model, name, X)` (frontend/views/prediction_utils.py) → deployed tree model
(RF/XGBoost) pe **TreeExplainer** chalata hai aur sirf is ek patient ke liye SHAP values nikaalta
hai. (ANN tree nahi, isliye SHAP sirf tree models pe.)

> Viva line: *"SHAP har patient ke liye real-time, model ke andar jhaank kar batata hai ke
> prediction kyun aaya — yeh sirf ek static chart nahi."*

---

## 6. Recommendations — `backend/recommendations/`

`generate_recommendations(probability, inputs, shap)` → rule-based clinical advice deta hai.
Risk ko 3 **tiers** mein baant-ta hai (`classify_risk`):

| Tier | Probability | Advice ka khulasa |
|------|------------|--------------------|
| **Low** | < 35% | Lifestyle/prevention, routine checkup |
| **Medium** | 35–55% | Zyada monitoring, 3–6 mahine mein lipid/BP test |
| **High** | > 55% | Urgent — cardiologist, ECG, stress test, lipid profile |

Saath hi patient-specific notes bhi add karta hai (e.g. agar BP ≥ 140, ya exercise angina ho).

---

## 7. RAG + AI Chatbot — `backend/rag/` aur `backend/llm/` ⭐ (Sabse zyada poocha jaata hai)

### 7.1 RAG kya hai?
**RAG = Retrieval-Augmented Generation.** Matlab: LLM se jawab dilwane se pehle, hum apni
**medical knowledge** mein se relevant tukde (chunks) dhoond kar (retrieve) LLM ko de dete hain,
taake jawab **grounded** ho (LLM apne pass se galat na bana de — "hallucination" kam ho).

**Simple steps:**
1. **Knowledge base** = `backend/knowledge_base/` ki `.md` files (AHA prevention, WHO CVD overview,
   lipid/BP reference, clinical features).
2. In files ko chote **chunks** (≈500 characters, 80 overlap) mein toda jaata hai.
3. Har chunk ko **embedding** (numbers ka vector) banaya jaata hai — model: `all-MiniLM-L6-v2`.
4. Yeh vectors **FAISS index** mein store hote hain (fast similarity search ke liye).
5. Jab user sawaal karta hai → sawaal ka bhi embedding banta hai → FAISS se **top 4 sabse milte
   chunks** nikalte hain → woh + patient context **Groq LLM** ko diye jaate hain → LLM jawab deta hai.

### 7.2 `rag.py` ke functions
- `collect_documents()` → **sirf** `knowledge_base/` ki files parhta hai aur chunks banata hai.
  (Pehle yeh `docs/` bhi parh raha tha — usse non-medical files chat mein aa rahi thi, ab fix.)
- `chunk_text(text, source)` → text ko 500-char chunks (80 overlap) mein toda.
- `_get_embedder()` → sentence-transformers model load (cached).
- `build_vector_store(force=False)` → **YEH WOH FUNCTION HAI JO "REBUILD" BUTTON CHALATA HAI.**
- `retrieve(query, top_k=4)` → sawaal ke sabse milte 4 chunks. (Agar index/embedder fail ho to
  `[]` return — app crash nahi karti, sirf LLM-only mode mein chali jaati hai.)
- `format_retrieved_context(chunks)` → chunks ko ek text block mein laata hai LLM ke liye.
- `rag_chat(message, patient_ctx, history)` → retrieve + patient context + Groq → final jawab.
- `rag_initial_explanation(patient_ctx)` → prediction ke baad pehla auto explanation.

### 7.3 **"Rebuild vector index" button daba'ne se kya hota hai?** (Pakka poocha jaayega)
Admin page pe yeh button `build_vector_store(force=True)` call karta hai. Steps:
1. `knowledge_base/` ki saari `.md` files dobara parhta hai.
2. Unko chunks mein torta hai.
3. Har chunk ka embedding (numbers) banata hai (`all-MiniLM-L6-v2`).
4. Ek naya **FAISS index** banata hai aur `data/vector_store/` mein save kar deta hai
   (purana overwrite).

**Kab daba'na hai?** Jab aap knowledge base mein koi nayi file add/edit karein — tab rebuild
karna parta hai taake chatbot nayi knowledge use kar sake. (`force=True` ka matlab: index pehle
se ho tab bhi naya bana do.)

### 7.4 `llm/groq_assistant.py` — LLM client
- `get_api_key()` → Groq key dhoondhta hai: pehle session (UI mein dali hui) → phir environment →
  phir Streamlit secrets. (UI key sirf us user ke session mein rehti hai — sab mein leak nahi hoti.)
- `chat_completion(messages, model)` → Groq API call (timeout ke saath). Models:
  `llama-3.3-70b-versatile` (default, best) ya `llama-3.1-8b-instant` (fast).
- `format_patient_context(ctx)` → patient ki prediction + SHAP ko ek structured text mein laata
  hai taake LLM ko poora pata ho. (System prompt LLM ko bolta hai: educational only, no prescriptions.)

---

## 8. Auth + Database — `backend/auth/` aur `backend/database/`

### `auth.py` (security)
- `hash_password` / `verify_password` → **bcrypt** se password hash (plain text kabhi save nahi).
- `register_user` → validation (username ≥3, valid email, password ≥6) + user banata hai.
- `login_user` → password check → ek random **session token** (`secrets.token_urlsafe`) banata hai,
  aur **7-din** ki expiry set karta hai.
- `resolve_session(token)` → token se user wapas laata hai (expired token reject).
- `logout_user(token)` → session delete.

### `database.py` (SQLite)
- `get_connection()` → DB connection (WAL mode + busy timeout — taake do log ek saath likhein to
  "database locked" na ho).
- Tables: **users, sessions, predictions, reports, chat_messages**.
- `save_prediction(...)` → ek prediction (inputs, ensemble, shap, recommendations) JSON ke roop
  mein save.
- `get_predictions / get_prediction` → history nikaalna.
- `save_report / get_report_for_prediction` → PDF ka path link karna.
- `save_chat_message / get_chat_history` → chatbot ki baat-cheet save/load.
- Har query **`WHERE user_id = ?`** se chalti hai — ek user dusre ka data nahi dekh sakta.

---

## 9. PDF Report — `backend/reporting/pdf_report.py`

`generate_pdf_report(...)` → **ReportLab** se hospital-style PDF banata hai:
header, patient inputs, ensemble result, SHAP factors, recommendations, AI explanation, disclaimer.
(AI/recommendation text ko `escape()` karta hai taake koi `< > &` symbol PDF ko crash na kare.)

---

## 10. Frontend — `frontend/` (jo screen pe dikhta hai)

- `streamlit_app.py` → **entry point**. Theme inject karta, session banata, auth check,
  `load_system()` (models cached), sidebar nav, aur page render.
- `ui/theme.py` → **poora design system** (colors, CSS). "Refined Clinical Blue" theme.
  Inputs/buttons/chat sab ko light + readable banata hai (Streamlit 1.58 ke `stBaseButton-*`,
  `stChatInput` waghaira selectors se).
- `ui/components.py` → reusable cards, badges, stat tiles, empty states.
- `ui/session.py` → session state ke default values.
- **Pages** (`frontend/views/`):
  - `auth_page.py` → login / signup (sidebar yahan hide hoti hai).
  - `home_page.py` → dashboard, KPIs, quick actions.
  - `prediction_page.py` → **13-field form** → prediction → 3 tabs (Model comparison /
    Recommendations / Explainability/SHAP) → PDF download.
  - `prediction_utils.py` → `build_features()` (inputs → 17 features, same engineering as training),
    `shap_for_instance()`, context builder.
  - `ai_assistant_page.py` → RAG chatbot UI.
  - `history_page.py` → purani assessments, dobara load karna, PDF download.
  - `analytics_page.py` → model metrics + training charts.
  - `admin_page.py` → DB counts + **"Rebuild vector index"** button.
  - `about_page.py` → project info.

> **Note:** folder ka naam `views/` hai (`pages/` nahi) — kyunki Streamlit `pages/` ko apna
> automatic menu bana leta tha. `views/` rakhne se woh extra (kharab) sidebar menu nahi banta.

---

## 11. Poora End-to-End Flow (user ka safar)

1. User **register/login** karta hai (bcrypt + session token).
2. **Risk Assessment** page pe 13 clinical values daalta hai.
3. `build_features()` → 17 features → `ensemble_predict()` → risk %.
4. **SHAP** → kaunse factors ne risk barhaya/ghataya.
5. **Recommendations** (low/medium/high tier).
6. **Groq + RAG** → AI explanation (knowledge-grounded).
7. Sab **SQLite** mein save + **PDF** ban-ta hai.
8. **Health Assistant** pe follow-up sawaal (RAG chatbot).
9. **History** pe purani report dobara dekh/download kar sakte hain.

---

## 12. Viva Q&A (rapid fire) 🎯

**Q: Aap ka project kya karta hai?**
A: Heart disease ka risk predict karta hai 3 ML models ke ensemble se, SHAP se explain karta hai,
recommendations + PDF deta hai, aur RAG-based AI assistant se sawaalon ke jawab deta hai. Educational hai.

**Q: Ensemble kyun, akela model kyun nahi?**
A: 3 models (RF, XGBoost, ANN) ka weighted average zyada robust hota hai — ek model ki ghalti
doosre balance kar lete hain. Weights ROC-AUC ke hisaab se.

**Q: Kaunsa model deploy hua aur kyun?**
A: Jo best ROC-AUC de woh (abhi Random Forest, ≈0.910). Tie hone pe XGBoost prefer kiya jaata hai
(0.005 margin). Yeh logic `train_and_deploy()` mein hai aur `model_metadata.json` mein record hota hai.

**Q: SHAP kya hai?**
A: Ek explainability technique — har feature ka contribution batati hai ke is patient ka risk
kyun itna aaya (positive = barhaya, negative = ghataya).

**Q: RAG kya hai aur kaise kaam karta hai?**
A: Retrieval-Augmented Generation. Knowledge base ko embeddings banakar FAISS mein store karte hain;
sawaal aane pe top relevant chunks retrieve karke Groq LLM ko dete hain — jawab grounded aur accurate.

**Q: Rebuild vector index button kya karta hai?**
A: `build_vector_store(force=True)` — knowledge base dobara parh kar, chunk, embed, aur FAISS index
naya bana deta hai. Knowledge base update karne ke baad chalana parta hai.

**Q: Frontend aur backend kaise connected hain?**
A: Ek hi process; frontend backend ko import karke functions call karta hai (no REST API). ML ka
model copy backend mein publish hota hai.

**Q: Data kitna aur kahan se?**
A: 4 public heart datasets → merge + dedup → ~1001 rows, ~55% disease, 17 features.

**Q: Security kaise?**
A: bcrypt password hashing, session tokens with 7-day expiry, har query user_id se isolated,
SQL parameterized (injection-safe), secrets `.env` mein (gitignored).

**Q: Limitations / future work?**
A: Educational only (PHI ke liye nahi); future: roles, rate-limiting, encryption-at-rest,
calibration, tests/CI, Docker.

---

## 13. Run karne ke commands (yaad rakho)

```powershell
# 1. Models train + backend mein publish (zaroorat ho to)
python machine_learning/run_pipeline.py

# 2. App chalao (project root se behtar; frontend se bhi chalega)
streamlit run frontend/streamlit_app.py

# Utilities
python backend/scripts/migrate_db.py                      # DB schema banao
python backend/scripts/build_knowledge_base.py --force    # RAG index rebuild (CLI se)
```

> Groq AI assistant ke liye `.env` mein `GROQ_API_KEY` daalna parta hai (ya UI ke settings box mein).
> Prediction, SHAP, PDF — yeh sab bina key ke bhi chalte hain.
