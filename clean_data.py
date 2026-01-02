import pandas as pd
import utils

pd.options.future.infer_string = True

df = pd.read_parquet(utils.FILE_BASE)

null_counts = df.isnull().sum()
# print(null_counts)
# print(LINE_SEPARATOR)

df.dropna(subset=["length_of_stay"], inplace=True)
print("Dropped rows with null length_of_stay")

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

df["length_of_stay"] = df["length_of_stay"].astype("int32")
print("Converted length_of_stay to int32")

# print(f'facility_id has decimal numbers: {utils.column_has_decimal_number(df, "facility_id")}')
# print(utils.show_decimal_values_in_column(df, 'facility_id'))
# print(utils.show_max_min_of_column(df, 'facility_id'))

df["facility_id"] = df["facility_id"].astype("Int32")
print("Converted facility_id to Int32")

# print(f'operating_certificate_number has decimal numbers: {utils.column_has_decimal_number(df, "operating_certificate_number")}')
# print(utils.show_decimal_values_in_column(df, 'operating_certificate_number'))
# print(utils.show_max_min_of_column(df, 'operating_certificate_number'))

df["operating_certificate_number"] = df["operating_certificate_number"].astype("Int32")
print("Converted operating_certificate_number to Int32")

columns = df.columns.to_list()

for col in columns:
    if df[col].dtype == "int64":
        max = df[col].max()
        min = df[col].min()
        if utils.INT_32_MIN <= min <= max <= utils.INT_32_MAX:
            df[col] = df[col].astype("Int32")
            print(f'Converted {col} to Int32')
'''Converted float64 type columns to Int32 where applicable'''
# df.to_parquet('data/updated_data_types.parquet', index=False)

'''Dropped 3 numeric columns that are not useful for prediction'''
df.drop(columns=['operating_certificate_number', 'facility_id', 'discharge_year'], axis=1, inplace=True)
print("Dropped operating_certificate_number, facility_id, discharge_year columns")
# df.to_parquet('data/numeric_feature_relation_analysis.parquet', index=False)

'''Dropped categorical description columns as we already have their corresponding code columns'''
df.drop(
    columns=[
        "ccs_diagnosis_description",
        "ccs_procedure_description",
        "apr_drg_description",
        "apr_mdc_description",
        "apr_severity_of_illness_description",
    ],
    axis=1,
    inplace=True,
)
print("Dropped categorical description columns")
# df.to_parquet('data/data_v3.parquet', index=False)

'''Convert total_charges and total_costs to int32 after removing $ sign and decimal part'''
# df['total_charges'] = df['total_charges'].replace(r'[\$,]', '', regex=True).astype('float64').astype('int32')
# df['total_costs'] = df['total_costs'].replace(r'[\$,]', '', regex=True).astype('float64').astype('int32')

# df.to_parquet('data/data_v4 .parquet', index=False)

# utils.show_max_min_of_column(df, 'total_charges')
# utils.show_max_min_of_column(df, 'total_costs')

'''Dropped total_costs and total_charges columns as these columns are calculated after discharge and a consequence of length_of_stay'''
df.drop(columns=['total_costs', 'total_charges'], axis=1, inplace=True)
print("Dropped total_costs and total_charges columns")

df.to_parquet('data/data_v5.parquet', index=False)

print(df.dtypes)