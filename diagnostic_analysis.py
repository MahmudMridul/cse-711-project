# diagnostic_analysis.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(csv_path):
    """Comprehensive dataset analysis"""
    df = pd.read_csv(csv_path)
    
    print("="*70)
    print("DATASET DIAGNOSTIC ANALYSIS")
    print("="*70)
    
    # 1. Check class distribution
    print("\n1. CLASS DISTRIBUTION:")
    print("-"*70)
    condition_counts = df['condition_type'].value_counts()
    print(condition_counts)
    print(f"\nClass imbalance ratio: {condition_counts.max() / condition_counts.min():.2f}")
    
    if condition_counts.max() / condition_counts.min() > 3:
        print("⚠ WARNING: Significant class imbalance detected!")
        print("  Solution: Use class_weight='balanced' in models")
    
    # 2. Check for data quality issues
    print("\n2. DATA QUALITY:")
    print("-"*70)
    print(f"Total records: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # 3. Feature correlation with target
    print("\n3. FEATURE IMPORTANCE (Correlation with Target):")
    print("-"*70)
    
    # Encode target
    le = LabelEncoder()
    df['condition_encoded'] = le.fit_transform(df['condition_type'])
    
    # Calculate correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corrwith(df['condition_encoded']).abs().sort_values(ascending=False)
    
    print("\nTop 10 most correlated features:")
    print(correlations.head(10))
    
    if correlations[1] < 0.1:  # Skip target itself
        print("\n⚠ WARNING: Low feature-target correlation!")
        print("  This suggests features may not be predictive")
        print("  Solution: Feature engineering needed")
    
    # 4. Hospital distribution
    print("\n4. HOSPITAL DISTRIBUTION:")
    print("-"*70)
    hospital_dist = df.groupby('hospital_name')['condition_type'].value_counts()
    print(hospital_dist.head(20))
    
    # 5. Temporal patterns
    print("\n5. TEMPORAL PATTERNS:")
    print("-"*70)
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    df['month'] = df['admission_date'].dt.month
    
    monthly_conditions = df.groupby(['month', 'condition_type']).size().unstack(fill_value=0)
    print(monthly_conditions)
    
    # 6. Feature statistics
    print("\n6. FEATURE STATISTICS:")
    print("-"*70)
    print(df.describe())
    
    return df

if __name__ == "__main__":
    df = analyze_dataset('data/data_clean.csv')