# 🏦 Lending Club Loan Defaulters Prediction

A machine learning project for predicting loan defaults using LendingClub data (2007-2018). This project applies Exploratory Data Analysis (EDA) and various ML models to identify patterns that indicate whether a person is likely to default on a loan.

## 📋 Table of Contents
- [Introduction](#introduction)
- [Business Objective](#business-objective)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Results](#models--results)
- [Technologies Used](#technologies-used)
- [License](#license)

## 📝 Introduction

**LendingClub** is a US peer-to-peer lending company and the world's largest peer-to-peer lending platform. This project aims to understand the driving factors behind loan defaults by analyzing historical loan data and building predictive models.

### Risk Types Addressed:
- **Business Loss Risk**: Not approving loans for applicants who would repay
- **Credit Loss Risk**: Approving loans for applicants who will default

## 🎯 Business Objective

- Identify risky loan applicants to reduce credit loss
- Understand driver variables behind loan defaults
- Support portfolio and risk assessment decisions
- Minimize financial losses while maximizing business opportunities

## 💾 Dataset

The project uses LendingClub loan data from 2007 to 2018 Q4.

📥 **Download Dataset**: [Kaggle - Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

| File | Description |
|------|-------------|
| `accepted_2007_to_2018Q4.csv` | Approved loan applications with outcomes |
| `rejected_2007_to_2018Q4.csv` | Rejected loan applications |

### Key Features:
- `loan_amnt` - Loan amount requested
- `term` - Loan term (36 or 60 months)
- `int_rate` - Interest rate
- `grade` / `sub_grade` - LC assigned loan grade
- `emp_length` - Employment length
- `home_ownership` - Home ownership status
- `annual_inc` - Annual income
- `dti` - Debt-to-income ratio
- `loan_status` - Current loan status (target variable)
- And many more...

## ️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "SME loan"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy hvplot jupyter
   ```

## 🚀 Usage

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the notebooks**
   - `main.ipynb` - For quick analysis and model comparison
   - `lending-club-loan-defaulters-prediction.ipynb` - For comprehensive EDA

3. **Run all cells** to execute the analysis

## 📊 Models & Results

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|-------|----------|---------|-----------|--------|----------|
| Logistic Regression | 98.23% | 99.44% | 91.55% | 95.21% | 93.34% |
| Random Forest | 98.83% | 99.31% | 99.65% | 91.36% | 95.33% |
| Gradient Boosting | 98.49% | 99.25% | 98.65% | 89.69% | 93.96% |

**Best Model**: Random Forest with 98.83% accuracy and 99.31% ROC-AUC score

## 🔧 Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning models
- **HvPlot** - Interactive visualizations
- **Jupyter Notebook** - Development environment

## 📈 Key Features Analyzed

- Loan amount and interest rates
- Employment length and income
- Debt-to-income ratio (DTI)
- FICO scores and risk assessment
- Geographic distribution
- Loan grades and purposes

## 📄 License

This project is for educational purposes as part of PDEU SEM 1 coursework.

---

⭐ If you found this project helpful, please give it a star!
