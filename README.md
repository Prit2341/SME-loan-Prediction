# 🏦 Loan Rejection Prediction System

A machine learning project to predict and understand **why loan applications get rejected vs accepted** using LendingClub data (2007-2018). Built with comprehensive EDA, feature analysis, and optimized Gradient Boosting model achieving **99.55% ROC-AUC**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
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
- [License](#-license)

---

## 🎯 Overview

This project analyzes Lending Club loan data to answer a critical business question:

> **Why do some loan applications get rejected while others get accepted?**

### Business Objectives
- Identify key factors that lead to loan rejection
- Build predictive models to classify applications
- Provide actionable insights for risk assessment
- Minimize credit losses while maximizing approvals

### Project Workflow
```
📂 Raw Data → 🔧 Preprocessing → 📊 EDA → 🔍 Feature Analysis → 🤖 ML Models → 🎯 Insights
```

---

## 🔍 Key Findings

### ❌ Why Loans Get REJECTED

| Factor | Impact | Description |
|--------|--------|-------------|
| **High DTI Ratio** | 🔴 Strong | Higher debt relative to income |
| **Low Credit Score** | 🔴 Strong | Poor credit history (FICO < 660) |
| **Short Employment** | 🟡 Moderate | Less job stability |
| **Large Loan Amount** | 🟢 Minor | Requesting too much |

### ✅ Why Loans Get ACCEPTED

| Factor | Impact | Description |
|--------|--------|-------------|
| **High Credit Score** | 🔴 Strong | FICO > 740 significantly helps |
| **Long Employment** | 🔴 Strong | 5+ years of stable employment |
| **Low DTI Ratio** | 🟡 Moderate | DTI < 35% preferred |
| **Reasonable Amount** | 🟢 Minor | Loan matches income level |

---

## 📁 Project Structure

```
SME loan/
│
├── � data/
│   ├── raw/                           # Original datasets
│   │   ├── accepted_2007_to_2018Q4.csv
│   │   └── rejected_2007_to_2018Q4.csv
│   └── processed/                     # Cleaned & processed data
│       ├── combined_loan_data.csv
│       ├── cleaned_accepted.csv
│       ├── cleaned_rejected.csv
│       └── model_ready_data.csv
│
├── 📂 models/
│   └── refined_gradient_boosting_model.joblib
│
├── 📂 notebooks/
│   └── EDA_loan_analysis.ipynb
│
├── 📂 results/
│   ├── figures/                       # Visualizations
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   ├── feature_importance.png
│   │   └── why_rejected_vs_accepted.png
│   └── metrics/                       # Model metrics & analysis
│       ├── model_results.csv
│       ├── best_model_parameters.csv
│       ├── why_rejected_vs_accepted.csv
│       └── feature_importance_ranking.csv
│
├── 📂 src/
│   ├── preprocess_data.py             # Data preprocessing pipeline
│   └── loan_rejection_model.py        # ML training & evaluation
│
├── 📄 .gitignore
├── 📄 README.md
├── 📄 requirements.txt
└── 📂 venv/                           # Virtual environment
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sme-loan-prediction.git
cd sme-loan-prediction
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
- Download from [Kaggle - Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Place CSV files in the project root directory

---

## 🚀 Usage

### 1. Preprocess Data
```bash
cd src
python preprocess_data.py
```
This creates files in `data/processed/`:
- `combined_loan_data.csv` - Merged accepted + rejected data
- `cleaned_accepted.csv` - Processed accepted loans
- `cleaned_rejected.csv` - Processed rejected loans

### 2. Run EDA (Jupyter Notebook)
```bash
jupyter notebook notebooks/EDA_loan_analysis.ipynb
```

### 3. Train & Evaluate Models
```bash
cd src
python loan_rejection_model.py
```
This will:
- Train 3 models (Logistic Regression, Random Forest, Gradient Boosting)
- Optimize the best model with hyperparameter tuning
- Generate visualizations and save results

### 4. Use the Trained Model
```python
import joblib
import pandas as pd

# Load the refined model
model = joblib.load('models/refined_gradient_boosting_model.joblib')

# Prepare new data (must have same features)
new_application = pd.DataFrame({
    'loan_amnt': [15000],
    'dti': [25.5],
    'emp_length': [5],
    'risk_score': [720]
})

# Predict
prediction = model.predict(new_application)
probability = model.predict_proba(new_application)[:, 1]

print(f"Prediction: {'Rejected' if prediction[0] == 1 else 'Accepted'}")
print(f"Rejection Probability: {probability[0]:.2%}")
```

---

## 📊 Model Performance

### Model Comparison

| Model | ROC-AUC | Accuracy | F1 Score | Precision | Recall |
|-------|---------|----------|----------|-----------|--------|
| **Gradient Boosting (Refined)** | **0.9955** | **98.04%** | **0.9758** | 99.09% | 96.11% |
| Gradient Boosting (Base) | 0.9933 | 96.74% | 0.9595 | 97.87% | 94.10% |
| Random Forest | 0.9844 | 94.65% | 0.9353 | 92.88% | 94.18% |
| Logistic Regression | 0.9428 | 86.42% | 0.8424 | 80.42% | 88.43% |

### Optimized Hyperparameters (Best Model)

| Parameter | Value |
|-----------|-------|
| n_estimators | 250 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.95 |
| min_samples_split | 5 |
| min_samples_leaf | 8 |

---

## 💾 Dataset

### Source
- **Kaggle**: [Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Period**: 2007 - 2018 Q4

### Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `accepted_2007_to_2018Q4.csv` | 2.26M | 151 | Approved loan applications |
| `rejected_2007_to_2018Q4.csv` | 27.6M | 9 | Rejected loan applications |
| `combined_loan_data.csv` | 3.58M | 71 | Merged & processed data |

### Key Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `loan_amnt` | Numeric | Loan amount requested ($) |
| `dti` | Numeric | Debt-to-income ratio (%) |
| `emp_length` | Numeric | Years of employment |
| `risk_score` | Numeric | Credit/FICO score |
| `addr_state_*` | Categorical | State (one-hot encoded) |
| `purpose_*` | Categorical | Loan purpose (one-hot encoded) |
| **`is_rejected`** | **Target** | **0 = Accepted, 1 = Rejected** |

---

## 🛠️ Technologies

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, Joblib |
| **Visualization** | Matplotlib, Seaborn |
| **Statistics** | SciPy |
| **Notebooks** | Jupyter |

---

## 📈 Visualizations

The project generates several visualizations:

- **ROC Curves** - Model performance comparison
- **Confusion Matrices** - Classification accuracy breakdown
- **Feature Importance** - Most predictive features
- **Probability Distributions** - Prediction confidence
- **Why Rejected vs Accepted** - Factor analysis chart

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@Prit2341](https://github.com/Prit2341)
- LinkedIn: [Prit Mayani](https://www.linkedin.com/in/prit-mayani-a35b371b9/)

---

## 🙏 Acknowledgments

- [LendingClub](https://www.lendingclub.com/) for the dataset
- [Kaggle](https://www.kaggle.com/) for hosting the data
- Scikit-learn documentation and community

---

<p align="center">
  Made with ❤️ for better loan decisions
</p>
