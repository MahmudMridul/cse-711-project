import pandas as pd
import data_info as info

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

file = 'data/combined_hospital_dataset_2020_2024.csv'

df = pd.read_csv(file)

# info.get_categorical_data_info(df)

days = df['admission_date'].str[:11].unique().tolist()


info.print_list(days)