"""
🏦 Loan Rejection Prediction Model
==================================
This script builds and evaluates ML models to predict loan rejection.
Uses the selected features from EDA analysis.

Author: Data Science Team
Date: 2024
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import joblib

# Settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Colors
COLORS = {'Accepted': '#2ecc71', 'Rejected': '#e74c3c'}

# Paths - relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')


def load_data():
    """Load the model-ready dataset with selected features."""
    print("=" * 70)
    print("📂 LOADING DATA")
    print("=" * 70)
    
    # Load model-ready data
    data_path = os.path.join(DATA_DIR, 'model_ready_data.csv')
    df = pd.read_csv(data_path)
    print(f"✅ Loaded: {data_path}")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Separate features and target
    X = df.drop('is_rejected', axis=1)
    y = df['is_rejected']
    
    print(f"\n📊 Target Distribution:")
    print(f"   Accepted (0): {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")
    print(f"   Rejected (1): {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
    
    return X, y


def prepare_data(X, y, test_size=0.2, sample_size=None):
    """Prepare data for modeling: split and scale."""
    print("\n" + "=" * 70)
    print("🔧 PREPARING DATA")
    print("=" * 70)
    
    # Sample if needed (for faster training)
    if sample_size and len(X) > sample_size:
        print(f"\n📊 Sampling {sample_size:,} records for faster training...")
        X_sampled, _, y_sampled, _ = train_test_split(
            X, y, train_size=sample_size, random_state=42, stratify=y
        )
        X, y = X_sampled, y_sampled
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train-Test Split:")
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✅ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    print("\n🔹 Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    print("🔹 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting model."""
    print("🔹 Training Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def refine_best_model(X_train, X_test, y_train, y_test, feature_names, scaler=None):
    """
    Refine the best model (Gradient Boosting) with hyperparameter tuning.
    """
    print("\n" + "=" * 70)
    print("🔧 REFINING BEST MODEL: GRADIENT BOOSTING")
    print("=" * 70)
    
    # Step 1: Hyperparameter Tuning with RandomizedSearchCV
    print("\n📊 Step 1: Hyperparameter Tuning...")
    print("   Using RandomizedSearchCV for faster search...")
    
    param_distributions = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.05, 0.08, 0.1, 0.12, 0.15],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [2, 4, 6, 8],
        'subsample': [0.8, 0.85, 0.9, 0.95, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Use a subset for faster tuning
    sample_size = min(50000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_tune = X_train[indices]
    y_tune = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
    
    print(f"   Tuning on {sample_size:,} samples...")
    
    gb_base = GradientBoostingClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        gb_base,
        param_distributions=param_distributions,
        n_iter=30,  # Number of random combinations to try
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_tune, y_tune)
    
    print(f"\n✅ Best Parameters Found:")
    for param, value in random_search.best_params_.items():
        print(f"   • {param}: {value}")
    print(f"\n   Best CV Score (ROC-AUC): {random_search.best_score_:.4f}")
    
    # Step 2: Train refined model on full training data
    print("\n📊 Step 2: Training Refined Model on Full Data...")
    
    refined_model = GradientBoostingClassifier(
        **random_search.best_params_,
        random_state=42
    )
    refined_model.fit(X_train, y_train)
    
    # Step 3: Evaluate refined model
    print("\n📊 Step 3: Evaluating Refined Model...")
    
    y_pred = refined_model.predict(X_test)
    y_prob = refined_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'Avg Precision': average_precision_score(y_test, y_prob)
    }
    
    print("\n" + "=" * 50)
    print("📊 REFINED MODEL PERFORMANCE")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"   {metric:<15}: {value:.4f}")
    
    # Step 4: Cross-validation for robust estimate
    print("\n📊 Step 4: Cross-Validation (5-Fold)...")
    
    cv_sample_size = min(100000, len(X_train))
    indices = np.random.choice(len(X_train), cv_sample_size, replace=False)
    X_cv = X_train[indices]
    y_cv = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
    
    cv_scores = cross_val_score(refined_model, X_cv, y_cv, cv=5, scoring='roc_auc', n_jobs=-1)
    
    print(f"\n   CV Scores: {cv_scores}")
    print(f"   Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Step 5: Feature importance from refined model
    print("\n📊 Step 5: Feature Importance (Refined Model)...")
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': refined_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n   Top 10 Most Important Features:")
    print("   " + "-" * 45)
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"   {i:2}. {row['Feature']:<30} {row['Importance']:.4f}")
    
    # Step 6: Save the refined model (and a self-contained Pipeline if scaler provided)
    print("\n📊 Step 6: Saving Refined Model...")

    model_path = os.path.join(MODELS_DIR, 'refined_gradient_boosting_model.joblib')
    joblib.dump(refined_model, model_path)
    print(f"   ✅ Saved: {model_path}")

    if scaler is not None:
        pipeline = Pipeline([('scaler', scaler), ('model', refined_model)])
        pipeline_path = os.path.join(MODELS_DIR, 'loan_rejection_pipeline.joblib')
        joblib.dump(pipeline, pipeline_path)
        print(f"   ✅ Saved: {pipeline_path} (scaler + model pipeline)")

    # Save best parameters
    params_df = pd.DataFrame([random_search.best_params_])
    params_path = os.path.join(METRICS_DIR, 'best_model_parameters.csv')
    params_df.to_csv(params_path, index=False)
    print(f"   ✅ Saved: {params_path}")
    
    # Save refined feature importance
    importance_path = os.path.join(METRICS_DIR, 'refined_feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"   ✅ Saved: {importance_path}")
    
    return refined_model, metrics, random_search.best_params_


def plot_refined_model_results(refined_model, X_test, y_test, feature_names):
    """Create visualizations for the refined model."""
    
    y_pred = refined_model.predict(X_test)
    y_prob = refined_model.predict_proba(X_test)[:, 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Confusion Matrix
    ax = axes[0, 0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Accepted', 'Rejected'],
                yticklabels=['Accepted', 'Rejected'],
                annot_kws={'size': 14})
    ax.set_title('Refined Model - Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    
    # Calculate and display rates
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    ax.text(0.5, -0.15, f'Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%}',
            transform=ax.transAxes, ha='center', fontsize=10)
    
    # 2. ROC Curve
    ax = axes[0, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    ax.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'Refined GB (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.3, color='#e74c3c')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Refined Model - ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 3. Feature Importance
    ax = axes[1, 0]
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': refined_model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(10)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance_df)))
    ax.barh(importance_df['Feature'], importance_df['Importance'], 
            color=colors, edgecolor='black')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top 10 Features - Refined Model', fontsize=14, fontweight='bold')
    
    # 4. Prediction Probability Distribution
    ax = axes[1, 1]
    accepted_probs = y_prob[y_test == 0]
    rejected_probs = y_prob[y_test == 1]
    
    ax.hist(accepted_probs, bins=50, alpha=0.6, label='Accepted', color='#2ecc71', edgecolor='black')
    ax.hist(rejected_probs, bins=50, alpha=0.6, label='Rejected', color='#e74c3c', edgecolor='black')
    ax.axvline(x=0.5, color='black', linestyle='--', lw=2, label='Threshold (0.5)')
    ax.set_xlabel('Predicted Probability of Rejection', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'refined_model_results.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: refined_model_results.png")


def print_refined_summary(metrics, best_params):
    """Print summary of the refined model."""
    print("\n" + "=" * 70)
    print("🏆 REFINED MODEL SUMMARY")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    🎯 REFINED GRADIENT BOOSTING MODEL               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │""")
    
    print(f"│  📊 PERFORMANCE METRICS                                            │")
    print(f"│     • Accuracy:      {metrics['Accuracy']:.4f}                                     │")
    print(f"│     • Precision:     {metrics['Precision']:.4f}                                     │")
    print(f"│     • Recall:        {metrics['Recall']:.4f}                                     │")
    print(f"│     • F1 Score:      {metrics['F1 Score']:.4f}                                     │")
    print(f"│     • ROC-AUC:       {metrics['ROC-AUC']:.4f}                                     │")
    print(f"│     • Avg Precision: {metrics['Avg Precision']:.4f}                                     │")
    
    print("""│                                                                     │
│  ⚙️  OPTIMIZED HYPERPARAMETERS                                      │""")
    
    for param, value in list(best_params.items())[:6]:
        print(f"│     • {param:<18}: {str(value):<20}              │")
    
    print("""│                                                                     │
│  📁 FILES SAVED                                                     │
│     • refined_gradient_boosting_model.joblib                        │
│     • best_model_parameters.csv                                     │
│     • refined_feature_importance.csv                                │
│     • refined_model_results.png                                     │
│                                                                     │
│  💡 USAGE                                                           │
│     model = joblib.load('refined_gradient_boosting_model.joblib')   │
│     predictions = model.predict(new_data)                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return metrics."""
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'Avg Precision': average_precision_score(y_test, y_prob)
    }
    
    return metrics, y_pred, y_prob


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate all models."""
    print("\n" + "=" * 70)
    print("🤖 TRAINING MODELS")
    print("=" * 70)
    
    models = {}
    results = []
    predictions = {}
    
    # 1. Logistic Regression
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train)
    metrics, y_pred, y_prob = evaluate_model(
        models['Logistic Regression'], X_test, y_test, 'Logistic Regression'
    )
    results.append(metrics)
    predictions['Logistic Regression'] = {'pred': y_pred, 'prob': y_prob}
    
    # 2. Random Forest
    models['Random Forest'] = train_random_forest(X_train, y_train)
    metrics, y_pred, y_prob = evaluate_model(
        models['Random Forest'], X_test, y_test, 'Random Forest'
    )
    results.append(metrics)
    predictions['Random Forest'] = {'pred': y_pred, 'prob': y_prob}
    
    # 3. Gradient Boosting
    models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train)
    metrics, y_pred, y_prob = evaluate_model(
        models['Gradient Boosting'], X_test, y_test, 'Gradient Boosting'
    )
    results.append(metrics)
    predictions['Gradient Boosting'] = {'pred': y_pred, 'prob': y_prob}
    
    print("✅ All models trained successfully!")
    
    return models, pd.DataFrame(results), predictions


def display_results(results_df):
    """Display model comparison results."""
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 70)
    
    # Format and display results
    results_display = results_df.copy()
    for col in results_display.columns[1:]:
        results_display[col] = results_display[col].apply(lambda x: f"{x:.4f}")
    
    print("\n" + results_display.to_string(index=False))
    
    # Find best model
    best_model = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
    best_auc = results_df['ROC-AUC'].max()
    
    print(f"\n🏆 Best Model: {best_model} (ROC-AUC: {best_auc:.4f})")
    
    return best_model


def plot_confusion_matrices(models, X_test, y_test):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Accepted', 'Rejected'],
                    yticklabels=['Accepted', 'Rejected'])
        axes[idx].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: confusion_matrices.png")


def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: roc_curves.png")


def plot_precision_recall_curves(models, X_test, y_test):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        
        ax.plot(recall, precision, color=color, lw=2, 
                label=f'{name} (AP = {ap:.4f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'precision_recall_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: precision_recall_curves.png")


def plot_feature_importance(models, feature_names, top_n=15):
    """Plot feature importance for tree-based models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Random Forest
    rf_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': models['Random Forest'].feature_importances_
    }).sort_values('Importance', ascending=True).tail(top_n)
    
    axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], 
                  color='#3498db', edgecolor='black')
    axes[0].set_title('Random Forest\nFeature Importance', fontweight='bold')
    axes[0].set_xlabel('Importance')
    
    # Gradient Boosting
    gb_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': models['Gradient Boosting'].feature_importances_
    }).sort_values('Importance', ascending=True).tail(top_n)
    
    axes[1].barh(gb_importance['Feature'], gb_importance['Importance'], 
                  color='#e74c3c', edgecolor='black')
    axes[1].set_title('Gradient Boosting\nFeature Importance', fontweight='bold')
    axes[1].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: feature_importance.png")


def plot_model_comparison_bar(results_df):
    """Plot bar chart comparing all metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    x = np.arange(len(metrics))
    width = 0.25
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics]
        ax.bar(x + i*width, values, width, label=row['Model'], color=colors[i], edgecolor='black')
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics]
        for j, v in enumerate(values):
            ax.text(x[j] + i*width, v + 0.01, f'{v:.3f}', ha='center', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: model_comparison.png")


def print_classification_reports(models, X_test, y_test):
    """Print classification reports for all models."""
    print("\n" + "=" * 70)
    print("📋 DETAILED CLASSIFICATION REPORTS")
    print("=" * 70)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{'='*50}")
        print(f"📊 {name}")
        print('='*50)
        print(classification_report(y_test, y_pred, 
                                    target_names=['Accepted', 'Rejected']))


def save_results(models, results_df, scaler, feature_names):
    """Save models and results."""
    print("\n" + "=" * 70)
    print("💾 SAVING RESULTS")
    print("=" * 70)

    # Save results to CSV
    results_path = os.path.join(METRICS_DIR, 'model_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✅ Saved: {results_path}")

    # Save feature names
    features_path = os.path.join(METRICS_DIR, 'model_features.csv')
    pd.DataFrame({'Feature': feature_names}).to_csv(features_path, index=False)
    print(f"✅ Saved: {features_path}")

    # Save scaler so new predictions can be correctly scaled
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved: {scaler_path}")

    print("\n📂 All results saved successfully!")


def analyze_why_rejected_vs_accepted(models, feature_names):
    """
    Analyze WHY loans get rejected vs accepted using model coefficients.
    This is the KEY insight - understanding the decision factors.
    """
    print("\n" + "=" * 70)
    print("🔍 WHY LOANS GET REJECTED vs ACCEPTED")
    print("=" * 70)
    
    # Use Logistic Regression coefficients (most interpretable)
    lr_model = models['Logistic Regression']
    coefficients = lr_model.coef_[0]
    
    # Create DataFrame with coefficients
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Separate into rejection factors and acceptance factors
    rejection_factors = coef_df[coef_df['Coefficient'] > 0].head(15)
    acceptance_factors = coef_df[coef_df['Coefficient'] < 0].head(15)
    
    # Print Rejection Factors
    print("\n" + "=" * 70)
    print("❌ WHY LOANS GET REJECTED (Positive Coefficients)")
    print("   Higher values in these features → MORE LIKELY TO BE REJECTED")
    print("=" * 70)
    
    for i, (_, row) in enumerate(rejection_factors.iterrows(), 1):
        feature = row['Feature']
        coef = row['Coefficient']
        
        # Interpret the feature
        if feature == 'dti':
            interpretation = "Higher debt-to-income ratio"
        elif feature == 'loan_amnt':
            interpretation = "Larger loan amounts requested"
        elif feature.startswith('addr_state_'):
            state = feature.replace('addr_state_', '')
            interpretation = f"Applicant from state: {state}"
        elif feature.startswith('purpose_'):
            purpose = feature.replace('purpose_', '').replace('_', ' ')
            interpretation = f"Loan purpose: {purpose}"
        elif feature == 'emp_length':
            interpretation = "Employment length impact"
        elif feature == 'risk_score':
            interpretation = "Higher risk score (unexpected - check data)"
        else:
            interpretation = feature
        
        print(f"   {i:2}. {interpretation:<45} (coef: {coef:+.4f})")
    
    # Print Acceptance Factors
    print("\n" + "=" * 70)
    print("✅ WHY LOANS GET ACCEPTED (Negative Coefficients)")
    print("   Higher values in these features → MORE LIKELY TO BE ACCEPTED")
    print("=" * 70)
    
    for i, (_, row) in enumerate(acceptance_factors.iterrows(), 1):
        feature = row['Feature']
        coef = row['Coefficient']
        
        # Interpret the feature
        if feature == 'risk_score':
            interpretation = "Higher credit/FICO score"
        elif feature == 'emp_length':
            interpretation = "Longer employment history"
        elif feature == 'loan_amnt':
            interpretation = "Moderate loan amount"
        elif feature == 'dti':
            interpretation = "Lower debt-to-income ratio"
        elif feature.startswith('addr_state_'):
            state = feature.replace('addr_state_', '')
            interpretation = f"Applicant from state: {state}"
        elif feature.startswith('purpose_'):
            purpose = feature.replace('purpose_', '').replace('_', ' ')
            interpretation = f"Loan purpose: {purpose}"
        else:
            interpretation = feature
        
        print(f"   {i:2}. {interpretation:<45} (coef: {coef:.4f})")
    
    # Save to CSV
    coef_df['Impact'] = coef_df['Coefficient'].apply(
        lambda x: 'REJECTION Factor' if x > 0 else 'ACCEPTANCE Factor'
    )
    coef_path = os.path.join(METRICS_DIR, 'why_rejected_vs_accepted.csv')
    coef_df.to_csv(coef_path, index=False)
    print(f"\n✅ Saved: {coef_path}")
    
    return coef_df


def plot_why_rejected_accepted(models, feature_names):
    """Visualize the rejection vs acceptance factors."""
    
    # Get coefficients from Logistic Regression
    lr_model = models['Logistic Regression']
    coefficients = lr_model.coef_[0]
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient')
    
    # Get top rejection and acceptance factors
    top_rejection = coef_df.tail(10)  # Highest positive = rejection
    top_acceptance = coef_df.head(10)  # Most negative = acceptance
    
    # Combine for visualization
    plot_df = pd.concat([top_acceptance, top_rejection])
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Coefficient bar chart
    ax = axes[0]
    colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in plot_df['Coefficient']]
    ax.barh(range(len(plot_df)), plot_df['Coefficient'], color=colors, edgecolor='black')
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['Feature'])
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Coefficient (← Acceptance | Rejection →)', fontsize=12)
    ax.set_title('Why Rejected vs Accepted\n(Logistic Regression Coefficients)', 
                 fontsize=14, fontweight='bold')
    
    # Add annotations
    ax.text(0.02, 0.98, '❌ REJECTION\nFactors', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', color='#e74c3c', fontweight='bold')
    ax.text(0.02, 0.02, '✅ ACCEPTANCE\nFactors', transform=ax.transAxes, 
            fontsize=10, verticalalignment='bottom', color='#2ecc71', fontweight='bold')
    
    # Plot 2: Feature importance from Random Forest
    ax = axes[1]
    rf_model = models['Random Forest']
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(15)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance_df)))
    ax.barh(importance_df['Feature'], importance_df['Importance'], 
            color=colors, edgecolor='black')
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Most Important Features\n(Random Forest)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'why_rejected_vs_accepted.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: why_rejected_vs_accepted.png")


def print_key_insights():
    """Print the key insights about rejection vs acceptance."""
    print("\n" + "=" * 70)
    print("📋 KEY INSIGHTS: WHY LOANS ARE REJECTED vs ACCEPTED")
    print("=" * 70)
    
    insights = """
┌─────────────────────────────────────────────────────────────────────┐
│                    🎯 MAIN REJECTION FACTORS                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 📉 LOW CREDIT/RISK SCORE                                       │
│     • Applicants with lower FICO scores are more likely rejected   │
│     • This is the STRONGEST predictor of rejection                 │
│                                                                     │
│  2. 💳 HIGH DEBT-TO-INCOME RATIO (DTI)                             │
│     • Higher DTI = More existing debt relative to income           │
│     • Lenders see this as higher default risk                      │
│                                                                     │
│  3. 💰 LOAN AMOUNT                                                  │
│     • Very large loan requests may signal higher risk              │
│     • Must be balanced with income and credit profile              │
│                                                                     │
│  4. 📍 GEOGRAPHIC LOCATION                                          │
│     • Some states have higher rejection rates                      │
│     • May reflect regional economic conditions                     │
│                                                                     │
│  5. 🎯 LOAN PURPOSE                                                 │
│     • Small business & educational loans: higher rejection         │
│     • Debt consolidation: more likely accepted                     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                    ✅ MAIN ACCEPTANCE FACTORS                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 📈 HIGH CREDIT/RISK SCORE                                       │
│     • Higher FICO scores strongly predict acceptance               │
│     • Shows history of responsible credit use                      │
│                                                                     │
│  2. 💼 LONGER EMPLOYMENT HISTORY                                    │
│     • More years of employment = more stable income                │
│     • Lower perceived default risk                                 │
│                                                                     │
│  3. 📉 LOW DEBT-TO-INCOME RATIO                                     │
│     • Lower DTI shows more capacity to repay                       │
│     • Indicates financial health                                   │
│                                                                     │
│  4. 🏦 REASONABLE LOAN AMOUNTS                                      │
│     • Moderate loan sizes relative to income                       │
│     • Shows realistic borrowing behavior                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""
    print(insights)


def print_summary(results_df, best_model):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("🎯 FINAL SUMMARY")
    print("=" * 70)
    
    summary = f"""
📊 MODEL TRAINING COMPLETE

Best Performing Model: {best_model}

Model Rankings by ROC-AUC:
"""
    print(summary)
    
    ranked = results_df.sort_values('ROC-AUC', ascending=False)
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        medal = ['🥇', '🥈', '🥉'][i-1]
        print(f"   {medal} {row['Model']}: ROC-AUC = {row['ROC-AUC']:.4f}, F1 = {row['F1 Score']:.4f}")
    
    print("""
📁 Files Generated:
   • model_results.csv - Performance metrics
   • model_features.csv - Feature list
   • why_rejected_vs_accepted.csv - Rejection/Acceptance factors
   • why_rejected_vs_accepted.png - Visualization of factors
   • confusion_matrices.png - Confusion matrices
   • roc_curves.png - ROC curves
   • precision_recall_curves.png - PR curves
   • feature_importance.png - Feature importance
   • model_comparison.png - Metrics comparison
""")


def main():
    """Main function to run the entire pipeline."""
    print("\n" + "=" * 70)
    print("🏦 LOAN REJECTION PREDICTION MODEL")
    print("=" * 70)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load data
    X, y = load_data()
    
    # Step 2: Prepare data (sample 500K for faster training)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
        X, y, test_size=0.2, sample_size=500000
    )
    
    # Step 3: Train and evaluate models
    models, results_df, predictions = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )
    
    # Step 4: Display results
    best_model = display_results(results_df)
    
    # Step 5: Print detailed reports
    print_classification_reports(models, X_test, y_test)
    
    # Step 6: Analyze WHY rejected vs accepted (KEY ANALYSIS!)
    print("\n" + "=" * 70)
    print("🔍 ANALYZING WHY REJECTED vs ACCEPTED")
    print("=" * 70)
    
    coef_df = analyze_why_rejected_vs_accepted(models, feature_names)
    plot_why_rejected_accepted(models, feature_names)
    print_key_insights()
    
    # Step 7: Generate additional visualizations
    print("\n" + "=" * 70)
    print("📈 GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_precision_recall_curves(models, X_test, y_test)
    plot_feature_importance(models, feature_names)
    plot_model_comparison_bar(results_df)
    
    # Step 8: Save results
    save_results(models, results_df, scaler, feature_names)
    
    # Step 9: Print summary
    print_summary(results_df, best_model)
    
    # Step 10: Refine the best model (Gradient Boosting)
    refined_model, refined_metrics, best_params = refine_best_model(
        X_train, X_test, y_train, y_test, feature_names, scaler=scaler
    )
    
    # Step 11: Visualize refined model results
    plot_refined_model_results(refined_model, X_test, y_test, feature_names)
    
    # Step 12: Print refined model summary
    print_refined_summary(refined_metrics, best_params)

    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return models, results_df, refined_model


if __name__ == "__main__":
    models, results_df, refined_model = main()
