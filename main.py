import pandas as pd
import data_info as info

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

file = 'data/hospital_admission_with_weather.csv'

df = pd.read_csv(file)
info.get_categorical_data_info(df)