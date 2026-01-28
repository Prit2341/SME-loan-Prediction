"""
Preprocessing Script for Lending Club Loan Data
Goal: Understand why loans are REJECTED vs ACCEPTED
Combines both datasets using common features
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Paths - relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def preprocess_accepted_loans(filepath=None):
    """
    Preprocess accepted loans - extract common features
    """
    if filepath is None:
        filepath = os.path.join(RAW_DATA_DIR, 'accepted_2007_to_2018Q4.csv')
    
    print("="*60)
    print("PREPROCESSING ACCEPTED LOANS DATA")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"   Original shape: {df.shape}")
    
    # Select only the common/comparable features
    print("\n2. Selecting common features...")
    
    # Extract common features that exist in both datasets
    df_clean = pd.DataFrame()
    
    # Loan amount
    df_clean['loan_amnt'] = df['loan_amnt']
    
    # Debt-to-Income ratio
    df_clean['dti'] = df['dti']
    
    # State
    df_clean['addr_state'] = df['addr_state']
    
    # Employment length - clean it
    df_clean['emp_length'] = df['emp_length'].astype(str)
    df_clean['emp_length'] = df_clean['emp_length'].str.replace('< 1 year', '0')
    df_clean['emp_length'] = df_clean['emp_length'].str.replace('10+ years', '10')
    df_clean['emp_length'] = df_clean['emp_length'].str.extract(r'(\d+)').astype(float)
    
    # Purpose/Title of loan
    df_clean['purpose'] = df['purpose'].astype(str).str.lower().str.strip()
    
    # Risk score - use FICO scores (average of low and high)
    df_clean['risk_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    
    # Target: 0 = Accepted
    df_clean['is_rejected'] = 0
    
    print(f"   Selected features: {df_clean.columns.tolist()}")
    print(f"   Shape: {df_clean.shape}")
    
    return df_clean


def preprocess_rejected_loans(filepath=None):
    """
    Preprocess rejected loans - extract common features
    """
    if filepath is None:
        filepath = os.path.join(RAW_DATA_DIR, 'rejected_2007_to_2018Q4.csv')
    
    print("\n" + "="*60)
    print("PREPROCESSING REJECTED LOANS DATA")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"   Original shape: {df.shape}")
    
    # Rename columns to match accepted data
    print("\n2. Renaming and selecting common features...")
    
    df_clean = pd.DataFrame()
    
    # Loan amount
    df_clean['loan_amnt'] = df['Amount Requested']
    
    # Debt-to-Income ratio - clean % sign
    df_clean['dti'] = df['Debt-To-Income Ratio'].astype(str).str.replace('%', '')
    df_clean['dti'] = pd.to_numeric(df_clean['dti'], errors='coerce')
    
    # State
    df_clean['addr_state'] = df['State']
    
    # Employment length - clean it
    df_clean['emp_length'] = df['Employment Length'].astype(str)
    df_clean['emp_length'] = df_clean['emp_length'].str.replace('< 1 year', '0')
    df_clean['emp_length'] = df_clean['emp_length'].str.replace('10+ years', '10')
    df_clean['emp_length'] = df_clean['emp_length'].str.extract(r'(\d+)').astype(float)
    
    # Purpose/Title of loan
    df_clean['purpose'] = df['Loan Title'].astype(str).str.lower().str.strip()
    
    # Risk score
    df_clean['risk_score'] = df['Risk_Score']
    
    # Target: 1 = Rejected
    df_clean['is_rejected'] = 1
    
    print(f"   Selected features: {df_clean.columns.tolist()}")
    print(f"   Shape: {df_clean.shape}")
    
    return df_clean


def combine_and_process(df_accepted, df_rejected, output_path=None):
    """
    Combine both datasets and perform final preprocessing
    """
    if output_path is None:
        output_path = os.path.join(PROCESSED_DATA_DIR, 'combined_loan_data.csv')
    
    print("\n" + "="*60)
    print("COMBINING DATASETS")
    print("="*60)
    
    # Sample rejected data to balance (rejected is much larger)
    print("\n1. Balancing datasets...")
    n_accepted = len(df_accepted)
    n_rejected = len(df_rejected)
    print(f"   Accepted: {n_accepted:,}")
    print(f"   Rejected: {n_rejected:,}")
    
    # Sample rejected to have reasonable ratio (e.g., 1:1 or 1:2)
    sample_size = min(n_rejected, n_accepted * 2)  # Max 2:1 ratio
    df_rejected_sampled = df_rejected.sample(n=sample_size, random_state=42)
    print(f"   Sampled rejected: {len(df_rejected_sampled):,}")
    
    # Combine datasets
    print("\n2. Combining datasets...")
    df = pd.concat([df_accepted, df_rejected_sampled], axis=0, ignore_index=True)
    print(f"   Combined shape: {df.shape}")
    
    # Clean purpose column - group similar purposes
    print("\n3. Cleaning purpose categories...")
    purpose_mapping = {
        'debt_consolidation': 'debt_consolidation',
        'debt consolidation': 'debt_consolidation',
        'credit_card': 'credit_card',
        'credit card refinancing': 'credit_card',
        'home_improvement': 'home_improvement',
        'home improvement': 'home_improvement',
        'major_purchase': 'major_purchase',
        'small_business': 'small_business',
        'business': 'small_business',
        'car': 'car',
        'medical': 'medical',
        'moving': 'moving',
        'vacation': 'vacation',
        'wedding': 'wedding',
        'house': 'house',
        'renewable_energy': 'renewable_energy',
        'educational': 'educational'
    }
    
    # Map known purposes, keep top 15 and group rest as 'other'
    top_purposes = df['purpose'].value_counts().nlargest(15).index.tolist()
    df['purpose'] = df['purpose'].apply(
        lambda x: purpose_mapping.get(x, x) if x in purpose_mapping else 
                  (x if x in top_purposes else 'other')
    )
    print(f"   Unique purposes: {df['purpose'].nunique()}")
    
    # Handle missing values
    print("\n4. Handling missing values...")
    print(f"   Missing before:\n{df.isnull().sum()}")
    
    # Fill numeric columns with median
    numeric_cols = ['loan_amnt', 'dti', 'emp_length', 'risk_score']
    for col in numeric_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical with mode or 'UNKNOWN'
    df['addr_state'].fillna('UNKNOWN', inplace=True)
    df['purpose'].fillna('other', inplace=True)
    
    print(f"   Missing after: {df.isnull().sum().sum()}")
    
    # Remove rows with any remaining nulls
    df.dropna(inplace=True)
    
    # Encode categorical variables
    print("\n5. Encoding categorical variables...")
    
    # One-hot encode categorical columns
    categorical_cols = ['addr_state', 'purpose']
    
    for col in categorical_cols:
        if col in df.columns:
            # Convert to string type to ensure proper encoding
            df[col] = df[col].astype(str)
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
    
    # Final summary
    print("\n6. Final dataset summary...")
    print(f"   Shape: {df.shape}")
    print(f"   Target distribution:")
    print(f"     Accepted (0): {(df['is_rejected']==0).sum():,}")
    print(f"     Rejected (1): {(df['is_rejected']==1).sum():,}")
    
    # Save combined data
    print(f"\n7. Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("   Done!")
    
    return df


def preprocess_accepted_only(filepath=None, output_path=None):
    """
    Full preprocessing of accepted loans for default prediction
    (separate model to understand why accepted loans default)
    """
    if filepath is None:
        filepath = os.path.join(RAW_DATA_DIR, 'accepted_2007_to_2018Q4.csv')
    if output_path is None:
        output_path = os.path.join(PROCESSED_DATA_DIR, 'cleaned_accepted.csv')
    
    print("\n" + "="*60)
    print("FULL PREPROCESSING - ACCEPTED LOANS (Default Prediction)")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"   Original shape: {df.shape}")
    
    # Remove columns with too many missing values (>50%)
    print("\n2. Removing columns with >50% missing values...")
    missing_pct = df.isnull().sum() / len(df) * 100
    cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"   Dropped {len(cols_to_drop)} columns")
    
    # Remove unnecessary columns
    print("\n3. Removing unnecessary columns...")
    unnecessary_cols = ['id', 'member_id', 'url', 'desc', 'title', 'emp_title', 
                        'zip_code', 'policy_code', 'application_type',
                        'funded_amnt', 'funded_amnt_inv', 'pymnt_plan',
                        'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                        'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
                        'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
                        'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
                        'last_fico_range_high', 'last_fico_range_low', 'hardship_flag',
                        'debt_settlement_flag', 'disbursement_method']
    cols_to_remove = [col for col in unnecessary_cols if col in df.columns]
    df.drop(columns=cols_to_remove, inplace=True, errors='ignore')
    
    # Filter and create target variable
    print("\n4. Creating target variable...")
    print(f"   Loan status distribution:\n{df['loan_status'].value_counts()}")
    
    # Keep only completed loans
    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(valid_statuses)]
    
    # Binary target: 0 = Paid, 1 = Default
    df['target'] = df['loan_status'].apply(lambda x: 0 if x == 'Fully Paid' else 1)
    df.drop(columns=['loan_status'], inplace=True)
    print(f"   After filtering: {len(df):,} rows")
    print(f"   Paid: {(df['target']==0).sum():,}, Default: {(df['target']==1).sum():,}")
    
    # Clean numeric columns
    print("\n5. Cleaning numeric columns...")
    
    if 'int_rate' in df.columns:
        df['int_rate'] = df['int_rate'].astype(str).str.replace('%', '').astype(float)
    
    if 'revol_util' in df.columns:
        df['revol_util'] = df['revol_util'].astype(str).str.replace('%', '')
        df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce')
    
    if 'term' in df.columns:
        df['term'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float)
    
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].astype(str)
        df['emp_length'] = df['emp_length'].str.replace('< 1 year', '0')
        df['emp_length'] = df['emp_length'].str.replace('10+ years', '10')
        df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
    
    # Handle date columns
    print("\n6. Processing date columns...")
    date_cols = ['issue_d', 'earliest_cr_line']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%b-%Y', errors='coerce')
            if df[col].notna().any():
                min_date = df[col].min()
                df[f'{col}_months'] = ((df[col] - min_date).dt.days / 30).astype(float)
            df.drop(columns=[col], inplace=True)
    
    # Encode categorical variables
    print("\n7. Encoding categorical variables...")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if df[col].nunique() <= 20:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
        else:
            df[col] = df[col].astype('category').cat.codes
        df.drop(columns=[col], inplace=True, errors='ignore')
    
    # Handle missing values
    print("\n8. Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    df.dropna(inplace=True)
    
    # Save
    print(f"\n9. Saving to {output_path}...")
    print(f"   Final shape: {df.shape}")
    df.to_csv(output_path, index=False)
    print("   Done!")
    
    return df


def preprocess_rejected_only(filepath=None, output_path=None):
    """
    Full preprocessing of rejected loans dataset
    """
    # Set default paths using organized folder structure
    if filepath is None:
        filepath = os.path.join(RAW_DATA_DIR, 'rejected_2007_to_2018Q4.csv')
    if output_path is None:
        output_path = os.path.join(PROCESSED_DATA_DIR, 'cleaned_rejected.csv')
    
    print("\n" + "="*60)
    print("FULL PREPROCESSING - REJECTED LOANS")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"   Original shape: {df.shape}")
    
    # Rename columns
    print("\n2. Renaming columns...")
    column_mapping = {
        'Amount Requested': 'loan_amnt',
        'Application Date': 'application_date',
        'Loan Title': 'purpose',
        'Risk_Score': 'risk_score',
        'Debt-To-Income Ratio': 'dti',
        'Zip Code': 'zip_code',
        'State': 'addr_state',
        'Employment Length': 'emp_length',
        'Policy Code': 'policy_code'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Drop unnecessary columns
    df.drop(columns=['zip_code', 'policy_code'], inplace=True, errors='ignore')
    
    # Clean DTI
    print("\n3. Cleaning numeric columns...")
    df['dti'] = df['dti'].astype(str).str.replace('%', '')
    df['dti'] = pd.to_numeric(df['dti'], errors='coerce')
    
    # Clean employment length
    df['emp_length'] = df['emp_length'].astype(str)
    df['emp_length'] = df['emp_length'].str.replace('< 1 year', '0')
    df['emp_length'] = df['emp_length'].str.replace('10+ years', '10')
    df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
    
    # Process date
    print("\n4. Processing date column...")
    if 'application_date' in df.columns:
        df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
        df['application_year'] = df['application_date'].dt.year
        df['application_month'] = df['application_date'].dt.month
        df.drop(columns=['application_date'], inplace=True)
    
    # Clean purpose
    print("\n5. Cleaning purpose categories...")
    df['purpose'] = df['purpose'].astype(str).str.lower().str.strip()
    top_purposes = df['purpose'].value_counts().nlargest(15).index.tolist()
    df['purpose'] = df['purpose'].apply(lambda x: x if x in top_purposes else 'other')
    
    # Encode categorical
    print("\n6. Encoding categorical variables...")
    for col in ['addr_state', 'purpose']:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
    
    # Handle missing
    print("\n7. Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    df.dropna(inplace=True)
    
    # Save
    print(f"\n8. Saving to {output_path}...")
    print(f"   Final shape: {df.shape}")
    df.to_csv(output_path, index=False)
    print("   Done!")
    
    return df


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LENDING CLUB DATA PREPROCESSING")
    print("Goal: Understand why loans are REJECTED vs ACCEPTED")
    print("="*60)
    
    # 1. Preprocess both datasets with common features
    df_accepted = preprocess_accepted_loans()
    df_rejected = preprocess_rejected_loans()
    
    # 2. Combine for rejection prediction model
    df_combined = combine_and_process(df_accepted, df_rejected)
    
    # 3. Also create cleaned individual datasets
    df_accepted_full = preprocess_accepted_only()
    df_rejected_full = preprocess_rejected_only()
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nOutput files (in data/processed/):")
    print("  1. combined_loan_data.csv - For predicting rejection vs acceptance")
    print("  2. cleaned_accepted.csv   - For predicting loan default (accepted loans only)")
    print("  3. cleaned_rejected.csv   - Cleaned rejected loans data")
    print("\nKey insights to explore:")
    print("  - What features lead to loan rejection?")
    print("  - How does DTI ratio affect acceptance?")
    print("  - Does employment length matter?")
    print("  - What is the role of risk score?")
    print("  - Which loan purposes get rejected more often?")
