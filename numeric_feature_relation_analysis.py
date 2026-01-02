import pandas as pd
import numpy as np
from scipy.stats import pearsonr, kendalltau, chi2_contingency
from sklearn.feature_selection import mutual_info_regression, f_regression
import utils


df = pd.read_parquet(utils.FILE_V5)

# Target column
target = df['length_of_stay']

# Get all Int32 columns (excluding target)
int32_columns = df.select_dtypes(include=['Int32', 'int32']).columns.tolist()
# print(int32_columns)

# Initialize results dictionary
results = {
    'Feature': [],
    'Pearson_Corr': [],
    'Pearson_PValue': [],
    'Kendall_Tau': [],
    'Kendall_PValue': [],
    'Mutual_Info': [],
    'ANOVA_F': [],
    'ANOVA_PValue': [],
    'Chi_Square': [],
    'Chi_Square_PValue': []
}

# Process each Int32 feature
for col in int32_columns:
    # Remove rows with NaN in either target or feature
    mask = df[col].notna() & target.notna()
    feature_clean = df.loc[mask, col]
    target_clean = target[mask]
    
    if len(feature_clean) == 0:
        continue
    
    results['Feature'].append(col)
    
    # 1. Pearson Correlation
    try:
        pearson_coef, pearson_p = pearsonr(feature_clean, target_clean)
        results['Pearson_Corr'].append(pearson_coef)
        results['Pearson_PValue'].append(pearson_p)
    except:
        results['Pearson_Corr'].append(np.nan)
        results['Pearson_PValue'].append(np.nan)
    
    # 2. Kendall's Tau
    try:
        kendall_coef, kendall_p = kendalltau(feature_clean, target_clean)
        results['Kendall_Tau'].append(kendall_coef)
        results['Kendall_PValue'].append(kendall_p)
    except:
        results['Kendall_Tau'].append(np.nan)
        results['Kendall_PValue'].append(np.nan)
    
    # 3. Mutual Information
    try:
        mi_score = mutual_info_regression(
            feature_clean.values.reshape(-1, 1), 
            target_clean, 
            random_state=42
        )[0]
        results['Mutual_Info'].append(mi_score)
    except:
        results['Mutual_Info'].append(np.nan)
    
    # 4. ANOVA F-statistic
    try:
        f_stat, f_p = f_regression(
            feature_clean.values.reshape(-1, 1), 
            target_clean
        )
        results['ANOVA_F'].append(f_stat[0])
        results['ANOVA_PValue'].append(f_p[0])
    except:
        results['ANOVA_F'].append(np.nan)
        results['ANOVA_PValue'].append(np.nan)
    
    # 5. Chi-Square Test (bin continuous variables)
    try:
        # Bin both variables into quartiles
        feature_binned = pd.qcut(feature_clean, q=4, labels=False, duplicates='drop')
        target_binned = pd.qcut(target_clean, q=4, labels=False, duplicates='drop')
        
        # Create contingency table
        contingency_table = pd.crosstab(feature_binned, target_binned)
        
        # Perform chi-square test
        chi2_stat, chi2_p, _, _ = chi2_contingency(contingency_table)
        results['Chi_Square'].append(chi2_stat)
        results['Chi_Square_PValue'].append(chi2_p)
    except:
        results['Chi_Square'].append(np.nan)
        results['Chi_Square_PValue'].append(np.nan)

# Create results DataFrame
results_df = pd.DataFrame(results)

# Sort by absolute Pearson correlation
results_df['Abs_Pearson'] = results_df['Pearson_Corr'].abs()
results_df = results_df.sort_values('Abs_Pearson', ascending=False)
results_df = results_df.drop('Abs_Pearson', axis=1)

# Display results
print("Relationship Analysis: length_of_stay vs Int32 Features\n")
print(results_df.to_string(index=False))

# Round to 5 decimal places before saving
results_df_rounded = results_df.round(5)

# Save to CSV
results_df_rounded.to_csv('data/numeric_feature_relation_analysis.csv', index=False)

# Summary of top features
print("\n" + "="*80)
print("TOP 5 FEATURES BY EACH METHOD:")
print("="*80)

print("\nBy Pearson Correlation (absolute):")
print(results_df.nlargest(5, 'Pearson_Corr', keep='all')[['Feature', 'Pearson_Corr']])

print("\nBy Mutual Information:")
print(results_df.nlargest(5, 'Mutual_Info')[['Feature', 'Mutual_Info']])

print("\nBy ANOVA F-statistic:")
print(results_df.nlargest(5, 'ANOVA_F')[['Feature', 'ANOVA_F']])

print("\nBy Chi-Square:")
print(results_df.nlargest(5, 'Chi_Square')[['Feature', 'Chi_Square']])
