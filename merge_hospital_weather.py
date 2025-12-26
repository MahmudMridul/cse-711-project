import pandas as pd

hospital_file = 'data/hospital_admission_no_timestamp.csv'
weather_file = 'data/weather_data.csv'

hos_df = pd.read_csv(hospital_file)
wea_df = pd.read_csv(weather_file)

merded_df = pd.merge(hos_df, wea_df, on='admission_date', how='left')

merded_df.to_csv('data/hospital_admission_weather_dataset.csv', index=False)