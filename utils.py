FILE_BASE = 'data/hospital_inpatient_discharges.parquet'
FILE_V1 = 'data/data_v1.parquet'
FILE_V2 = 'data/data_v2.parquet'
FILE_V3 = 'data/data_v3.parquet'
LINE_SEPARATOR = '-' * 50
INT_32_MAX = 2_147_483_647
INT_32_MIN = -2_147_483_648


def print_list(list):
    for item in list:
        print(item)
    print(LINE_SEPARATOR)

def column_has_decimal_number(df, column_name):
    return (df[column_name] % 1 != 0).any()

def show_decimal_values_in_column(df, column_name):
    rows_with_decimal = df[df[column_name] % 1 != 0][column_name]
    print(rows_with_decimal)
    print(f'Number of rows with decimal values in {column_name}: {len(rows_with_decimal)}')
    print(LINE_SEPARATOR)

def show_max_min_of_column(df, column_name):
    max_value = df[column_name].max()
    min_value = df[column_name].min()
    print(f'Max {column_name}: {max_value}')
    print(f'Min {column_name}: {min_value}')
    print(LINE_SEPARATOR)