import pandas as pd
import joblib

# ------------------ LOAD SCALER ------------------
scaler_filename = 'scaler.joblib'
loaded_scaler = joblib.load(scaler_filename)

print(f"✅ Scaler loaded from '{scaler_filename}'")

# ------------------ DEFINE NUMERICAL COLUMNS ------------------
numerical_cols = ['survey_year', 'age']

# ------------------ PREPROCESS FUNCTION ------------------
def preprocess_input(data_dict, reference_columns):
    input_df = pd.DataFrame([data_dict])

    # Handle gender mapping
    if 'gender' in input_df.columns:
        input_df['gender'] = input_df['gender'].replace({
            'Other': 'Other / Prefer not to say'
        })

    # ------------------ SCALING ------------------
    for col in numerical_cols:
        if col not in input_df.columns:
            raise ValueError(f"Missing numerical column: {col}")

    input_df[numerical_cols] = loaded_scaler.transform(input_df[numerical_cols])

    # ------------------ ONE HOT ENCODING ------------------
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # ------------------ ALIGN COLUMNS ------------------
    # Add missing columns
    for col in reference_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Remove extra columns
    input_encoded = input_encoded[reference_columns]

    return input_encoded


# ------------------ EXAMPLE INPUT ------------------
sample_survey_response = {
    'survey_year': 2023,
    'age': 30,
    'gender': 'Male',
    'country': 'United States',
    'self_employed': 'No',
    'family_history': 'Yes',
    'work_interfere': 'Sometimes',
    'no_employees': '26-100',
    'remote_work': 'Yes',
    'tech_company': 'Yes',
    'benefits': 'Yes',
    'care_options': 'Yes',
    'wellness_program': 'No',
    'seek_help': 'Yes',
    'anonymity': 'Yes',
    'leave': 'Somewhat easy',
    'mental_health_consequence': 'No',
    'phys_health_consequence': 'No',
    'coworkers': 'Yes',
    'supervisor': 'Yes',
    'mental_health_interview': 'No',
    'phys_health_interview': 'No',
    'mental_vs_physical': 'Yes',
    'obs_consequence': 'No'
}

# ------------------ LOAD REFERENCE COLUMNS ------------------
reference_feature_columns = joblib.load("columns.pkl")

# ------------------ TEST ------------------
preprocessed_sample = preprocess_input(
    sample_survey_response,
    reference_feature_columns
)

print("✅ Preprocessed sample input:")
print(preprocessed_sample.head())

print("Shape:", preprocessed_sample.shape)
print("Columns match:", list(preprocessed_sample.columns) == reference_feature_columns)
