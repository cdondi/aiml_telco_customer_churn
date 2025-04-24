# 🧭 Project Roadmap: Customer Churn Prediction

This roadmap guides the end-to-end development of a customer churn prediction system incorporating real-world AI/ML interview concepts, data streaming, and production-ready workflows.

---

## 🔰 Phase 1: Project Setup & Environment

- Scaffold folder structure: `data/`, `notebooks/`, `models/`, `sql/`, `app/`, `utils/`
- Create and activate Conda or virtualenv environment
- Install dependencies via `pip install -r requirements.txt`
- Include: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `datasets`, `joblib`, `matplotlib`, `seaborn`, `imbalanced-learn`

---

## 📦 Phase 2: Dataset Preparation

- Load data from `data/telco7k.csv`
- Use DVC with 2 versions of data:
  - One dataset simulates 1 million rows by duplicating, adding noise and shuffling the rows for streaming and metrics comparison
    - I also want to run experiments and observe the effects of duplicate data on model performance
  - The other dataset has the original 7K rows.
- Duplicate with noise for streaming simulation
- Perform basic EDA and save `data/raw_7K.csv`
- Perform basic EDA and save `data/raw_1M_streaming.csv`

---

## 🧹 Phase 3: Data Cleaning & Preprocessing

- Handle missing values with `fillna`, `dropna`, or `interpolate`
- Encode categorical variables (label or one-hot)
- Normalize numerics (e.g., `MinMaxScaler`)
- Save cleaned output to `data/processed.csv`

---

## 🧪 Phase 4: Streaming Simulation & Feature Engineering

- Read data in chunks using `pandas.read_csv(chunksize=...)`
- Fit encoders/scalers on first chunk and reuse
- Engineer features: tenure buckets, % usage drop, etc.

---

## 🧠 Phase 5: Modeling (Incremental and Full)

- Train `SGDClassifier` using `.partial_fit()` for streamed training
- Optionally train full-batch model (e.g., `XGBoost`) for comparison
- Evaluate with F1, Precision, Recall, ROC-AUC

---

## ⚖️ Phase 6: Class Imbalance Handling

- Apply `class_weight='balanced'` or resample
- Monitor recall and F1 on minority class (churn)

---

## 💾 Phase 7: Model Persistence

- Save models with `joblib.dump()`
- Create batch prediction pipeline using saved model

---

## 📊 Phase 8: Visualization & Reporting

- Plot confusion matrix, ROC/PR curves, and feature importances
- Save outputs to `reports/` directory

---

## 🚀 Phase 9: Deployment (Optional)

- Create Streamlit UI or FastAPI endpoint for prediction
- Package model, encoders, and logic for reuse

---

## 📈 Phase 10: Review & Interview Readiness

- Write detailed `README.md` for GitHub
- Map tasks to interview topics (feature engineering, streaming, evaluation)
- Practice discussing project and tradeoffs

---

## 📌 Bonus: Stretch Goals

- Add DVC for version control
- Build Makefile or YAML pipeline
- Add test coverage for pipeline components
