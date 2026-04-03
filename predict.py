"""
Loan Rejection Predictor
========================
Run a quick prediction on a new loan application using the trained pipeline.

Usage:
    python predict.py

The pipeline (scaler + Gradient Boosting model) handles all preprocessing
internally — just pass raw feature values.
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, 'models', 'loan_rejection_pipeline.joblib')


def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Pipeline not found at {PIPELINE_PATH}.\n"
            "Run: cd src && python loan_rejection_model.py"
        )
    return joblib.load(PIPELINE_PATH)


def predict(loan_amnt, dti, emp_length, risk_score):
    """
    Predict whether a loan application will be rejected.

    Parameters
    ----------
    loan_amnt  : float  — Loan amount requested ($)
    dti        : float  — Debt-to-income ratio (%)
    emp_length : float  — Years of employment
    risk_score : float  — FICO / credit score

    Returns
    -------
    dict with keys: prediction, probability, verdict
    """
    pipeline = load_pipeline()

    application = pd.DataFrame({
        'loan_amnt':  [loan_amnt],
        'dti':        [dti],
        'emp_length': [emp_length],
        'risk_score': [risk_score],
    })

    prediction  = pipeline.predict(application)[0]
    probability = pipeline.predict_proba(application)[0]

    return {
        'prediction':         int(prediction),
        'rejection_prob':     float(probability[1]),
        'acceptance_prob':    float(probability[0]),
        'verdict':            'REJECTED' if prediction == 1 else 'ACCEPTED',
    }


def print_result(result, loan_amnt, dti, emp_length, risk_score):
    verdict_icon = '❌' if result['verdict'] == 'REJECTED' else '✅'
    print("\n" + "=" * 55)
    print("  🏦 LOAN APPLICATION PREDICTION")
    print("=" * 55)
    print(f"  Loan Amount   : ${loan_amnt:,.0f}")
    print(f"  DTI Ratio     : {dti}%")
    print(f"  Employment    : {emp_length} years")
    print(f"  Risk Score    : {risk_score}")
    print("-" * 55)
    print(f"  Verdict       : {verdict_icon}  {result['verdict']}")
    print(f"  Rejection Prob: {result['rejection_prob']:.2%}")
    print(f"  Acceptance Prob: {result['acceptance_prob']:.2%}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    # --- Example applications ---

    print("\n--- Example 1: High-risk applicant ---")
    r1 = predict(loan_amnt=35000, dti=42.0, emp_length=1, risk_score=580)
    print_result(r1, 35000, 42.0, 1, 580)

    print("--- Example 2: Low-risk applicant ---")
    r2 = predict(loan_amnt=10000, dti=18.5, emp_length=8, risk_score=760)
    print_result(r2, 10000, 18.5, 8, 760)

    print("--- Example 3: Borderline applicant ---")
    r3 = predict(loan_amnt=20000, dti=31.0, emp_length=3, risk_score=670)
    print_result(r3, 20000, 31.0, 3, 670)
