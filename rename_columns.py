import pandas as pd
import utils

file = 'data/hospital_inpatient_discharges.parquet'
df = pd.read_parquet(file)

print(f'Number of rows: {len(df)}')

df.rename(columns=str.lower, inplace=True)
df.rename(columns={'zip code - 3 digits': 'zip code'}, inplace=True)
df.columns = df.columns.str.replace(' ', '_')

columns = df.columns.to_list()
utils.print_list(columns)

df.to_parquet('data/hospital_inpatient_discharges.parquet', index=False)