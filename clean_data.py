import pandas as pd
import utils
import numpy as np
from scipy.stats import pointbiserialr, spearmanr, chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression

pd.options.future.infer_string = True

FILE = 'data/hospital_inpatient_discharges.parquet'
LINE_SEPARATOR = '-' * 50
INT_32_MAX = 2_147_483_647
INT_32_MIN = -2_147_483_648

df = pd.read_parquet(FILE)

null_counts = df.isnull().sum()
# print(null_counts)
# print(LINE_SEPARATOR)

df.dropna(subset=['length_of_stay'], inplace=True)

# null_counts = temp_df.isnull().sum()
# print(null_counts)
# print(LINE_SEPARATOR)

# print(f'Rows: {len(df)}')
# print(f'Rows after dropping nulls in length_of_stay: {len(temp_df)}')
# print(LINE_SEPARATOR)

# columns = df.columns.to_list()
# utils.print_list(columns)
# print(len(df))

# max_length_of_stay = df['length_of_stay'].max()
# min_length_of_stay = df['length_of_stay'].min()
# print(max_length_of_stay)
# print(min_length_of_stay)

df['length_of_stay'] = df['length_of_stay'].astype('int32')

# print(f'facility_id has decimal numbers: {utils.column_has_decimal_number(df, "facility_id")}')
# print(utils.show_decimal_values_in_column(df, 'facility_id'))
# print(utils.show_max_min_of_column(df, 'facility_id'))

df['facility_id'] = df['facility_id'].astype('Int32')

# print(f'operating_certificate_number has decimal numbers: {utils.column_has_decimal_number(df, "operating_certificate_number")}')
# print(utils.show_decimal_values_in_column(df, 'operating_certificate_number'))
# print(utils.show_max_min_of_column(df, 'operating_certificate_number'))

df['operating_certificate_number'] = df['operating_certificate_number'].astype('Int32')

columns = df.columns.to_list()

for col in columns:
    if df[col].dtype == 'int64':
        max = df[col].max()
        min = df[col].min()
        if INT_32_MIN <= min <= max <= INT_32_MAX:
            df[col] = df[col].astype('Int32')

# print(df.dtypes)
