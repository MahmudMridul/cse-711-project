import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load your dataset
df = pd.read_csv('data/data_clean.csv')

# Define target column
target = 'admission_count'

# Exclude non-predictive columns
exclude_cols = ['admission_date', 'hospital_name', target]

# Separate categorical and numerical features
categorical_features = ['patient_age_group', 'patient_gender', 'severity_level', 'seasonal_indicator']
numerical_features = [col for col in df.columns if col not in exclude_cols + categorical_features]

print("="*80)
print("FEATURE IMPORTANCE ANALYSIS FOR PREDICTING CONDITION_TYPE")
print("="*80)

# ============================================================================
# METHOD 1: CHI-SQUARE TEST (for categorical features)
# ============================================================================
print("\n" + "="*80)
print("METHOD 1: CHI-SQUARE TEST & CRAMÉR'S V (Categorical Features)")
print("="*80)
print("Interpretation: Higher values indicate stronger association with target")
print("-"*80)

chi_square_results = []

for feature in categorical_features:
    if feature in df.columns:
        # Create contingency table
        contingency_table = pd.crosstab(df[feature], df[target])
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramér's V (effect size)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
        cramers_v = np.sqrt(chi2_stat / (n * min_dim))
        
        chi_square_results.append({
            'Feature': feature,
            'Chi-Square': round(chi2_stat, 4),
            'P-Value': round(p_value, 6),
            'Cramers_V': round(cramers_v, 4)
        })

chi_df = pd.DataFrame(chi_square_results).sort_values('Cramers_V', ascending=False)
print("\n", chi_df.to_string(index=False))
print("\nNote: Cramér's V ranges from 0 (no association) to 1 (perfect association)")

# ============================================================================
# METHOD 2: ANOVA F-STATISTIC (for numerical features)
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: ANOVA F-STATISTIC & ETA-SQUARED (Numerical Features)")
print("="*80)
print("Interpretation: Higher F-stat means feature better separates target classes")
print("-"*80)

anova_results = []

# Get groups for each condition type
groups = df.groupby(target)

for feature in numerical_features:
    if feature in df.columns:
        # Prepare groups for ANOVA
        group_data = [group[feature].dropna() for name, group in groups]
        
        # Perform one-way ANOVA
        if len(group_data) > 1 and all(len(g) > 0 for g in group_data):
            f_stat, p_value = f_oneway(*group_data)
            
            # Calculate Eta-squared (effect size)
            # SS_between / SS_total
            grand_mean = df[feature].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_data)
            ss_total = ((df[feature] - grand_mean)**2).sum()
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            anova_results.append({
                'Feature': feature,
                'F-Statistic': round(f_stat, 4),
                'P-Value': round(p_value, 6),
                'Eta-Squared': round(eta_squared, 4)
            })

anova_df = pd.DataFrame(anova_results).sort_values('F-Statistic', ascending=False)
print("\n", anova_df.to_string(index=False))
print("\nNote: Eta-squared ranges from 0 (no effect) to 1 (complete effect)")

# ============================================================================
# METHOD 3: MUTUAL INFORMATION (for all features)
# ============================================================================
print("\n" + "="*80)
print("METHOD 3: MUTUAL INFORMATION (All Features)")
print("="*80)
print("Interpretation: Measures information gain; higher = more predictive power")
print("-"*80)

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(df[target])

# Prepare features
X = df[categorical_features + numerical_features].copy()

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Calculate mutual information
mi_scores = mutual_info_classif(X, y, discrete_features=[col in categorical_features for col in X.columns], random_state=42)

mi_results = pd.DataFrame({
    'Feature': X.columns,
    'Mutual_Information': np.round(mi_scores, 4),
    'Type': ['Categorical' if col in categorical_features else 'Numerical' for col in X.columns]
}).sort_values('Mutual_Information', ascending=False)

print("\n", mi_results.to_string(index=False))
print("\nNote: Higher MI means feature shares more information with target")

# ============================================================================
# METHOD 4: CORRELATION RATIO (Eta) - Alternative measure
# ============================================================================
print("\n" + "="*80)
print("METHOD 4: CORRELATION RATIO (ETA) - Numerical Features")
print("="*80)
print("Interpretation: Measures proportion of variance explained (0 to 1)")
print("-"*80)

def correlation_ratio(categories, values):
    """Calculate correlation ratio (eta)"""
    categories = np.array(categories)
    values = np.array(values)
    
    # Remove NaN values
    mask = ~(pd.isna(categories) | pd.isna(values))
    categories = categories[mask]
    values = values[mask]
    
    # Calculate category means
    cat_unique = np.unique(categories)
    cat_means = [values[categories == cat].mean() for cat in cat_unique]
    cat_counts = [np.sum(categories == cat) for cat in cat_unique]
    
    # Overall mean
    mean_y = values.mean()
    
    # Between group variance
    ssb = sum(count * (cat_mean - mean_y)**2 for count, cat_mean in zip(cat_counts, cat_means))
    
    # Total variance
    sst = sum((values - mean_y)**2)
    
    # Correlation ratio
    if sst == 0:
        return 0
    return np.sqrt(ssb / sst)

eta_results = []

for feature in numerical_features:
    if feature in df.columns:
        eta = correlation_ratio(df[target], df[feature])
        eta_results.append({
            'Feature': feature,
            'Eta (Correlation_Ratio)': round(eta, 4)
        })

eta_df = pd.DataFrame(eta_results).sort_values('Eta (Correlation_Ratio)', ascending=False)
print("\n", eta_df.to_string(index=False))

# ============================================================================
# SUMMARY: TOP FEATURES BY EACH METHOD
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: TOP 10 MOST IMPORTANT FEATURES")
print("="*80)

# Combine all results
summary = []

# Add categorical features
for _, row in chi_df.iterrows():
    summary.append({
        'Feature': row['Feature'],
        'Type': 'Categorical',
        'Score': row['Cramers_V'],
        'Metric': 'Cramers_V'
    })

# Add numerical features (using Eta-squared from ANOVA)
for _, row in anova_df.iterrows():
    summary.append({
        'Feature': row['Feature'],
        'Type': 'Numerical',
        'Score': row['Eta-Squared'],
        'Metric': 'Eta-Squared'
    })

summary_df = pd.DataFrame(summary).sort_values('Score', ascending=False)
print("\n", summary_df.head(10).to_string(index=False))

# ============================================================================
# ADDITIONAL: STATISTICAL SIGNIFICANCE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE SUMMARY (P-Value < 0.05)")
print("="*80)

significant_features = []

# Categorical features
for _, row in chi_df.iterrows():
    if row['P-Value'] < 0.05:
        significant_features.append({
            'Feature': row['Feature'],
            'Type': 'Categorical',
            'P-Value': row['P-Value'],
            'Significant': 'Yes'
        })

# Numerical features
for _, row in anova_df.iterrows():
    if row['P-Value'] < 0.05:
        significant_features.append({
            'Feature': row['Feature'],
            'Type': 'Numerical',
            'P-Value': row['P-Value'],
            'Significant': 'Yes'
        })

sig_df = pd.DataFrame(significant_features).sort_values('P-Value')
print("\n", sig_df.to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nRECOMMENDATIONS:")
print("1. Focus on features with high Cramér's V (categorical) or Eta-squared (numerical)")
print("2. Consider features with high Mutual Information scores")
print("3. Use statistically significant features (p-value < 0.05)")
print("4. Consider using feature selection methods like RFE or LASSO for modeling")
print("="*80)