import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

file_1 = 'data/Hospital_Dataset_2020_2024.csv'
file_2 = 'data/Hospital_Dataset_2022_2024.csv'


df1 = pd.read_csv(file_1)
# df1 = pd.read_csv(file_1, parse_dates=['admission_date'])
df2 = pd.read_csv(file_2)
# df2 = pd.read_csv(file_2, parse_dates=['admission_date'])

# overlapping_dates = df1[df1['admission_date'].isin(df2['admission_date'])]['admission_date'].unique()

# if len(overlapping_dates) > 0:
#     print(f"Warning: Found {len(overlapping_dates)} overlapping dates!")
#     print(overlapping_dates)
# else:
#     print("Success: No overlapping dates found.")

common = pd.merge(df1, df2, how='inner')
print(f'{len(common)} common rows found between the two datasets.')
