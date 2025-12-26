def get_categorical_data_info(df):
    hospitals = df['hospital_name'].unique().tolist()
    condition_types = df['condition_type'].unique().tolist()
    age_groups = df['patient_age_group'].unique().tolist()
    seasons = df['seasonal_indicator'].unique().tolist()
    diag_codes = df['primary_diagnosis_code'].unique().tolist()
    severity_levels = df['severity_level'].unique().tolist()

    print(f'\nNumber of unique hospitals: {len(hospitals)}')
    print('Hospitals:')
    print_list(hospitals)

    print(f'\nNumber of unique condition types: {len(condition_types)}')
    print('Condition Types:')
    print_list(condition_types)

    print(f'\nNumber of unique age groups: {len(age_groups)}')
    print('Age Groups:')
    print_list(age_groups)

    print(f'\nNumber of unique seasonal indicators: {len(seasons)}')
    print('Seasonal Indicators:')
    print_list(seasons)

    print(f'\nNumber of unique primary diagnosis codes: {len(diag_codes)}')
    print('Primary Diagnosis Codes:')
    print_list(diag_codes)

    print(f'\nNumber of unique severity levels: {len(severity_levels)}')
    print('Severity Levels:')
    print_list(severity_levels)

    return {
        'hospitals': hospitals,
        'condition_types': condition_types,
        'age_groups': age_groups,
        'seasons': seasons,
        'diagnosis_codes': diag_codes,
        'severity_levels': severity_levels
    }


def print_list(list):
    for item in list:
        print(item)