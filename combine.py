import pandas as pd

file_1 = 'data/Hospital_Dataset_2020_2024.csv'
file_2 = 'data/Hospital_Dataset_2022_2024.csv'

df_1 = pd.read_csv(file_1)
df_2 = pd.read_csv(file_2)

combined_df = pd.concat([df_1, df_2], ignore_index=True)
combined_df.to_csv('data/combined_hospital_dataset_2020_2024.csv', index=False)