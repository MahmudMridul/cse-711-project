import pandas as pd
import data_info as info

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

file = 'data/data_clean.csv'

"""
Federated Learning Based Prediction of Hospital Patient Length of Stay.
"""

df = pd.read_csv(file)
# hospital_count = df['hospital_name'].value_counts()
# age_group = df['patient_age_group'].value_counts()
# print(hospital_count)

column = df.columns.to_list()
info.print_list(column)