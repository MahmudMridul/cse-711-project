import utils
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression, f_regression
import warnings
warnings.filterwarnings('ignore')

pd.options.future.infer_string = True

df = pd.read_parquet(utils.FILE_V5)

# print(df["birth_weight"].isna().sum())

# num = df["apr_severity_of_illness_code"].nunique()
# value_count = df["apr_severity_of_illness_code"].value_counts()
# print(num)
# print(utils.LINE_SEPARATOR)
# print(value_count)

# print(utils.LINE_SEPARATOR)
# print(utils.LINE_SEPARATOR)

# num = df["apr_severity_of_illness_description"].nunique()
# value_count = df["apr_severity_of_illness_description"].value_counts()
# print(num)
# print(utils.LINE_SEPARATOR)
# print(value_count)

# print(utils.LINE_SEPARATOR)
# print(df.dtypes)

# Identify categorical columns (string type)
categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
target = 'length_of_stay'

# Remove rows where target is null
df_clean = df[df[target].notna()].copy()

print(f"Analyzing {len(categorical_cols)} categorical columns\n")
print("="*80)

# Store results
results = []

for col in categorical_cols:
    # Skip if too many nulls
    if df_clean[col].isna().sum() / len(df_clean) > 0.5:
        continue
    
    # Remove nulls for this column
    temp_df = df_clean[[col, target]].dropna()
    
    if len(temp_df) < 10:
        continue
    
    # Method 1: ANOVA F-statistic (most common)
    groups = [group[target].values for name, group in temp_df.groupby(col)]
    if len(groups) > 1:
        f_stat, p_value = stats.f_oneway(*groups)
    else:
        f_stat, p_value = 0, 1
    
    # Method 2: Eta-squared (effect size for ANOVA)
    grand_mean = temp_df[target].mean()
    ss_between = sum(len(group) * (group[target].mean() - grand_mean)**2 
                     for name, group in temp_df.groupby(col))
    ss_total = sum((temp_df[target] - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    # Method 3: Mutual Information
    le = LabelEncoder()
    encoded = le.fit_transform(temp_df[col].astype(str))
    mi_score = mutual_info_regression(encoded.reshape(-1, 1), 
                                      temp_df[target].values, 
                                      random_state=42)[0]
    
    # Method 4: Correlation Ratio (Eta)
    eta = np.sqrt(eta_squared)
    
    # Store results
    results.append({
        'Column': col,
        'F_Statistic': f_stat,
        'P_Value': p_value,
        'Eta_Squared': eta_squared,
        'Eta': eta,
        'Mutual_Info': mi_score,
        'Unique_Values': temp_df[col].nunique()
    })

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Eta_Squared', ascending=False)

print("\nRESULTS (Sorted by Eta-Squared - Effect Size)")
print("="*80)
print(results_df.to_string(index=False))

print("\n" + "="*80)
print("\nINTERPRETATION GUIDE:")
print("-" * 80)
print("1. F-Statistic: Higher = stronger relationship (ANOVA)")
print("2. P-Value: < 0.05 = statistically significant")
print("3. Eta-Squared: 0.01=small, 0.06=medium, 0.14=large effect")
print("4. Eta (Correlation Ratio): 0-1, higher = stronger relationship")
print("5. Mutual Info: Higher = more information shared with target")
print("="*80)

# Top 5 most relevant columns
print("\n\nTOP 5 MOST RELEVANT COLUMNS:")
print("-" * 80)
for i, row in results_df.head(5).iterrows():
    print(f"{row['Column']:40s} | Eta²={row['Eta_Squared']:.4f} | MI={row['Mutual_Info']:.4f}")

# Prepare data for CSV export (matching the format of your example)
csv_results = []

for _, row in results_df.iterrows():
    csv_results.append({
        'Feature': row['Column'],
        'Pearson_Corr': np.nan,  # Not applicable for categorical variables
        'Pearson_PValue': np.nan,
        'Kendall_Tau': np.nan,
        'Kendall_PValue': np.nan,
        'Mutual_Info': row['Mutual_Info'],
        'ANOVA_F': row['F_Statistic'],
        'ANOVA_PValue': row['P_Value'],
        'Eta_Squared': row['Eta_Squared'],
        'Eta': row['Eta'],
        'Unique_Values': row['Unique_Values']
    })

# Create CSV dataframe
csv_df = pd.DataFrame(csv_results)

# Sort by Eta_Squared (descending)
csv_df = csv_df.sort_values('Eta_Squared', ascending=False)

# Write to CSV
output_file = 'data/categorical_feature_relation_analysis.csv'
csv_df = csv_df.round(5)
csv_df.to_csv(output_file, index=False)

print(f"\n\nResults saved to: {output_file}")
print("="*80)