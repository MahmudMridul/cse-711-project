import pandas as pd
import data_info as info

pd.set_option("display.max_columns", 5)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)

# file = "data/hospital_admission_with_weather.csv"

# snow = df[
#     "snowfall_sum_cm"
# ].value_counts()  # all rows contain zero, so drop this column
# df = df.drop(["snowfall_sum_cm"], axis=1)
# df.to_csv("data/hospital_admission_with_weather_v2.csv", index=False)


# file = "data/hospital_admission_with_weather_v2.csv"
# df = pd.read_csv(file)

# precipitation = df[
#     "precipitation_sum_mm"
# ].value_counts()  # almost all rows contain zero, so drop.
# print(precipitation)
# df = df.drop(["precipitation_sum_mm"], axis=1)
# df.to_csv("data/hospital_admission_with_weather_v3.csv", index=False)


# file = "data/data_drop_precipitation_v3.csv"
# df = pd.read_csv(file)

# rain = df["rain_sum_mm"].value_counts() # almost all rows contain zero, so drop.
# print(rain)
# df = df.drop(["rain_sum_mm"], axis=1)
# df.to_csv("data/data_drop_rain_v4.csv", index=False)


# file = "data/data_drop_rain_v4.csv"
# df = pd.read_csv(file)
# precipitation_h = df[
#     "precipitation_hours_h"
# ].value_counts()  # almost all rows contain zero, so drop.
# print(precipitation_h)
# df = df.drop(["precipitation_hours_h"], axis=1)
# df.to_csv("data/data_drop_precipitation_v5.csv", index=False)

# file = "data/data_drop_precipitation_v5.csv"
# df = pd.read_csv(file)
# print(len(df))
# df = df.drop(["primary_diagnosis_code"], axis=1)
# df.to_csv("data/data_drop_diagnosis_code_v6.csv", index=False)

file = "data/data_drop_diagnosis_code_v6.csv"
df = pd.read_csv(file)
df = df.drop(["length_of_stay_avg"], axis=1)
df.to_csv("data/data_drop_stay_v7.csv", index=False)
