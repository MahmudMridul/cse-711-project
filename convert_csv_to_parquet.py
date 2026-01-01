import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

file = "data/hospital_inpatient_discharges.csv"

df = pd.read_csv(file, low_memory=False, dtype={'Length of Stay': str})

df['Length of Stay'] = pd.to_numeric(df['Length of Stay'], errors='coerce')

table = pa.Table.from_pandas(df)

pq.write_table(
    table,
    "data/hospital_inpatient_discharges.parquet",
    compression="zstd",
    compression_level=19,
    use_dictionary=True,
    write_statistics=True,
)
