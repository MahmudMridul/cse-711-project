import pandas as pd
import utils as u
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

pd.options.future.infer_string = True

# df_base = pd.read_parquet(u.FILE_BASE)

df = pd.read_parquet(u.FILE_V8)

print("Columns and Data Types:")
print(df.dtypes)
print(u.LINE_SEPARATOR)

num_of_columns = len(df.columns)

# print(f"Number of Columns: {num_of_columns}")
# print(u.LINE_SEPARATOR)

number_of_rows = len(df)

# print(f"Number of Rows: {number_of_rows}")
# print(u.LINE_SEPARATOR)

# facility_unique_count = df["facility_name"].nunique()
# facility_name_count = df["facility_name"].value_counts()

# print("Facility Name Value Counts:")
# print(facility_name_count)
# print(u.LINE_SEPARATOR)

# print(f"Number of Unique Facilites: {facility_unique_count}")
# print(u.LINE_SEPARATOR)

# hospital_county_counts = df['hospital_county'].value_counts()
# unique_hospital_counties = df['hospital_county'].nunique()

# print("Hospital County Value Counts:")
# print(hospital_county_counts)
# print(u.LINE_SEPARATOR)

# print(f"Number of Unique Hospital Counties: {unique_hospital_counties}")
# print(u.LINE_SEPARATOR)

# patient_disposition_counts = df['patient_disposition'].value_counts()
# print("Patient Disposition Value Counts:")
# print(patient_disposition_counts)
# print(u.LINE_SEPARATOR)

# birth_weight_counts = df['birth_weight'].value_counts(dropna=False)
# print("Birth Weight Value Counts (including NaN):")
# print(birth_weight_counts)
# print(u.LINE_SEPARATOR)

# birth_weight_nulls = df['birth_weight'].isna().sum()
# print(f"Number of Nulls in Birth Weight: {birth_weight_nulls}")
# print(u.LINE_SEPARATOR)

# unique_birth_weights = df['birth_weight'].nunique(dropna=True)
# print(f"Number of Unique Birth Weights (excluding NaN): {unique_birth_weights}")
# print(u.LINE_SEPARATOR)

# print("Full list of Birth Weights:")
# print(df['birth_weight'].unique())
# print(u.LINE_SEPARATOR)

# payment_typology_1_null_count = df['payment_typology_1'].isna().sum()
# print(f"Number of Nulls in Payment Typology 1: {payment_typology_1_null_count}")
# print(f"Percentage of Nulls in Payment Typology 1: {(payment_typology_1_null_count / number_of_rows) * 100:.2f}%")
# print(u.LINE_SEPARATOR)

# payment_typology_2_null_count = df['payment_typology_2'].isna().sum()
# print(f"Number of Nulls in Payment Typology 2: {payment_typology_2_null_count}")
# print(f"Percentage of Nulls in Payment Typology 2: {(payment_typology_2_null_count / number_of_rows) * 100:.2f}%")
# print(u.LINE_SEPARATOR)

# payment_typology_3_null_count = df['payment_typology_3'].isna().sum()
# print(f"Number of Nulls in Payment Typology 3: {payment_typology_3_null_count}")
# print(f"Percentage of Nulls in Payment Typology 3: {(payment_typology_3_null_count / number_of_rows) * 100:.2f}%")
# print(u.LINE_SEPARATOR)
