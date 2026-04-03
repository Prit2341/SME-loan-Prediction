# 🏦 Loan Rejection Prediction System

A machine learning project that predicts **whether a loan application will be rejected or accepted** — and explains *why* — using real LendingClub data (2007–2018). Achieved **99.55% ROC-AUC** with a refined Gradient Boosting model.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-99.55%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Technologies](#-technologies)
- [Author](#-author)

---

## 🎯 Overview

Banks reject millions of loan applications without explaining why. This project solves that by building a system that:

> **Predicts loan rejection AND explains the key reasons behind it**

### Business Objectives
- Identify the key factors that cause loan rejection
- Build and compare multiple ML models for classification
- Provide actionable insights for applicants and lenders
- Deploy a self-contained prediction pipeline

### Project Workflow
```
Raw Data → Preprocessing → EDA → Feature Selection → Model Training → Hyperparameter Tuning → Pipeline
```

---

## 🔍 Key Findings

### Feature Importance (Refined Gradient Boosting)

| Rank | Feature | Importance | Role |
|------|---------|------------|------|
| 1 | `emp_length` | **74.15%** | Acceptance factor |
| 2 | `risk_score` | **20.80%** | Acceptance factor |
| 3 | `dti` | **4.24%** | Rejection factor |
| 4 | `loan_amnt` | **0.81%** | Minor rejection factor |

> **Key insight:** Employment length is a stronger predictor than credit score. Job stability matters more than past credit history to LendingClub's model.

### ❌ Why Loans Get REJECTED

| Factor | Logistic Coef | Description |
|--------|--------------|-------------|
| **High DTI Ratio** | +41.93 | Debt payments eat up most of income |
| **Large Loan Amount** | +0.09 | Requesting beyond means |

### ✅ Why Loans Get ACCEPTED

| Factor | Logistic Coef | Description |
|--------|--------------|-------------|
| **Long Employment** | -2.01 | Stable income = lower default risk |
| **High Credit Score** | -1.70 | History of responsible credit use |

---

## 📁 Project Structure

```
SME loan/
│
├── 📂 data/
│   ├── raw/                               # Original datasets (not tracked — too large)
│   │   ├── accepted_2007_to_2018Q4.csv    # 2.26M rows, 151 columns
│   │   └── rejected_2007_to_2018Q4.csv    # 27.6M rows, 9 columns
│   └── processed/                         # Cleaned & processed data (not tracked)
│       ├── combined_loan_data.csv          # Merged dataset (3.58M rows, 71 cols)
│       ├── model_ready_data.csv            # Final model input (4 features + target)
│       ├── cleaned_accepted.csv            # Accepted loans for default prediction
│       └── cleaned_rejected.csv            # Cleaned rejected loans
│
├── 📂 models/                             # Saved model artifacts (not tracked)
│   ├── loan_rejection_pipeline.joblib     # ⭐ Scaler + model combined (use this)
│   ├── refined_gradient_boosting_model.joblib
│   └── scaler.joblib                      # Standalone StandardScaler
│
├── 📂 notebooks/
│   └── EDA_loan_analysis.ipynb            # Exploratory data analysis
│
├── 📂 results/
│   ├── figures/                           # All visualizations (PNG)
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   ├── precision_recall_curves.png
│   │   ├── feature_importance.png
│   │   ├── model_comparison.png
│   │   ├── refined_model_results.png
│   │   └── why_rejected_vs_accepted.png
│   └── metrics/                           # Model metrics & analysis (CSV)
│       ├── model_results.csv
│       ├── best_model_parameters.csv
│       ├── refined_feature_importance.csv
│       ├── feature_importance_ranking.csv
│       ├── model_features.csv
│       ├── selected_features.txt
│       └── why_rejected_vs_accepted.csv
│
├── 📂 src/
│   ├── preprocess_data.py                 # Full data preprocessing pipeline
│   └── loan_rejection_model.py            # ML training, tuning & evaluation
│
├── 📄 predict.py                          # ⭐ Quick prediction demo script
├── 📄 .gitignore
├── 📄 README.md
└── 📄 requirements.txt
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- ~35GB disk space for raw data

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/Prit2341/SME-loan-Prediction.git
cd SME-loan-Prediction
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
- Download from [Kaggle — LendingClub Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Place both CSV files inside `data/raw/`

---

## 🚀 Usage

### Step 1 — Preprocess Data
```bash
python src/preprocess_data.py
```
Creates in `data/processed/`:
- `combined_loan_data.csv` — merged accepted + rejected (3.58M rows)
- `model_ready_data.csv` — 4 features + target, ready for training
- `cleaned_accepted.csv` — for default prediction (separate task)
- `cleaned_rejected.csv` — cleaned rejected loans

### Step 2 — Run EDA (optional)
```bash
jupyter notebook notebooks/EDA_loan_analysis.ipynb
```

### Step 3 — Train & Evaluate Models
```bash
# Windows (required for emoji print statements)
PYTHONIOENCODING=utf-8 python src/loan_rejection_model.py
```
This will:
- Train 3 models: Logistic Regression, Random Forest, Gradient Boosting
- Tune the best model with RandomizedSearchCV (30 iterations, 3-fold CV)
- Save `models/loan_rejection_pipeline.joblib` (scaler + model)
- Generate all visualizations and metrics

### Step 4 — Run Prediction Demo
```bash
python predict.py
```
Shows predictions for 3 example applicants (high risk / low risk / borderline).

### Step 5 — Use in Your Own Code
```python
import joblib
import pandas as pd

# Load self-contained pipeline — no manual scaling needed
pipeline = joblib.load('models/loan_rejection_pipeline.joblib')

new_application = pd.DataFrame({
    'loan_amnt':  [15000],
    'dti':        [25.5],
    'emp_length': [5],
    'risk_score': [720]
})

prediction  = pipeline.predict(new_application)
probability = pipeline.predict_proba(new_application)[:, 1]

print(f"Verdict:            {'REJECTED' if prediction[0] == 1 else 'ACCEPTED'}")
print(f"Rejection Probability: {probability[0]:.2%}")
```

---

## 📊 Model Performance

### Model Comparison

| Model | ROC-AUC | Accuracy | F1 Score | Precision | Recall |
|-------|---------|----------|----------|-----------|--------|
| **Gradient Boosting (Refined)** | **0.9955** | **98.04%** | **0.9758** | **99.09%** | **96.11%** |
| Gradient Boosting (Base) | 0.9933 | 96.74% | 0.9595 | 97.87% | 94.10% |
| Random Forest | 0.9844 | 94.65% | 0.9353 | 92.88% | 94.18% |
| Logistic Regression | 0.9428 | 86.42% | 0.8424 | 80.42% | 88.43% |

### 5-Fold Cross-Validation (Refined Model)
```
CV Scores : [0.9954, 0.9959, 0.9959, 0.9958, 0.9955]
Mean      : 0.9957
Std Dev   : ±0.0002  ← very stable, no overfitting
```

### Optimized Hyperparameters

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| `n_estimators` | 250 | Number of trees |
| `max_depth` | 6 | Tree complexity |
| `learning_rate` | 0.1 | Step size per tree |
| `subsample` | 0.95 | Data fraction per tree |
| `min_samples_split` | 5 | Min samples to split a node |
| `min_samples_leaf` | 8 | Min samples in a leaf |

---

## 💾 Dataset

### Source
- **Kaggle**: [LendingClub Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Period**: 2007 — 2018 Q4

### Raw Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `accepted_2007_to_2018Q4.csv` | 2.26M | 151 | Approved loan applications |
| `rejected_2007_to_2018Q4.csv` | 27.6M | 9 | Rejected loan applications |

### Class Imbalance Handling
Rejected loans outnumber accepted 12:1. Rejected data was **randomly sampled at a 2:1 ratio** vs accepted to balance classes without losing information.

| Split | Count |
|-------|-------|
| Accepted | 2,113,648 |
| Rejected (sampled) | 1,470,316 |
| **Total** | **3,583,964** |

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `loan_amnt` | Numeric | Loan amount requested ($) |
| `dti` | Numeric | Debt-to-income ratio (%) |
| `emp_length` | Numeric | Years of employment (0–10) |
| `risk_score` | Numeric | FICO credit score |
| **`is_rejected`** | **Target** | **0 = Accepted, 1 = Rejected** |

> `addr_state` and `purpose` were one-hot encoded during preprocessing but dropped in the final model (only top 4 features selected via combined RF + mutual information score).

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Pipeline, GradientBoosting, RandomForest, LogisticRegression) |
| **Model Persistence** | Joblib |
| **Visualization** | Matplotlib, Seaborn |
| **Statistics** | SciPy |
| **Notebooks** | Jupyter |

---

## 📈 Visualizations Generated

| File | Description |
|------|-------------|
| `roc_curves.png` | ROC curves for all 3 models |
| `confusion_matrices.png` | Confusion matrix per model |
| `precision_recall_curves.png` | Precision-Recall tradeoff curves |
| `feature_importance.png` | RF vs GB feature importance comparison |
| `model_comparison.png` | Side-by-side metric bar chart |
| `refined_model_results.png` | Refined model: CM, ROC, feature importance, probability distribution |
| `why_rejected_vs_accepted.png` | Logistic Regression coefficient analysis |

---

## 👤 Author

**Prit Mayani**
- GitHub: [@Prit2341](https://github.com/Prit2341)
- LinkedIn: [Prit Mayani](https://www.linkedin.com/in/prit-mayani-a35b371b9/)

---

## 🙏 Acknowledgments

- [LendingClub](https://www.lendingclub.com/) for the dataset
- [Kaggle](https://www.kaggle.com/) for hosting the data
- Scikit-learn community

---

<p align="center">
  Made with ❤️ for better loan decisions
</p>
