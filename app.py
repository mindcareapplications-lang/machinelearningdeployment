import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# 1. Load the trained model and label encoder
# Ensure these files are in the same directory as app.py
try:
    model = joblib.load('random_forest_model.joblib')
    le = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    print("Error: Model or LabelEncoder files not found. Please export them from Colab first.")

# 2. Define the exact columns used during model training
# This is CRITICAL to ensure the model receives data in the correct format/order
MODEL_COLUMNS = [
    'respondent_id', 'survey_year', 'age', 'gender_Male',
    'gender_Non-binary', 'gender_Other / Prefer not to say',
    'country_Brazil', 'country_Canada', 'country_France', 'country_Germany',
    'country_India', 'country_Mexico', 'country_Other',
    'country_United Kingdom', 'country_United States',
    'country_colombia', 'self_employed_Yes', 'family_history_Yes',
    'work_interfere_Often', 'work_interfere_Rarely',
    'work_interfere_Sometimes', 'no_employees_100-500',
    'no_employees_26-100', 'no_employees_500-Jan', 'no_employees_6-25',
    'no_employees_More than 1000', 'remote_work_Yes', 'tech_company_Yes',
    'benefits_No', 'benefits_Not sure', 'care_options_No',
    'care_options_Not sure', 'wellness_program_No',
    'wellness_program_Not sure', 'seek_help_No', 'seek_help_Not sure',
    'anonymity_Yes', 'anonymity_Not sure', "leave_Don't know",
    'leave_Somewhat difficult', 'leave_Somewhat easy', 'leave_Very easy',
    'mental_health_consequence_No', 'mental_health_consequence_Yes',
    'phys_health_consequence_No', 'phys_health_consequence_Yes',
    'coworkers_Some of them', 'coworkers_Yes',
    'supervisor_Some of them', 'supervisor_Yes',
    'mental_health_interview_No', 'mental_health_interview_Yes',
    'phys_health_interview_No', 'phys_health_interview_Yes',
    'mental_vs_physical_No', 'mental_vs_physical_Yes',
    'obs_consequence_Yes'
]

# Categorical columns that need one-hot encoding
CATEGORICAL_COLS = [
    'gender', 'country', 'self_employed', 'family_history', 'work_interfere',
    'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
    'wellness_program', 'seek_help', 'anonymity', 'leave',
    'mental_health_consequence', 'phys_health_consequence', 'coworkers',
    'supervisor', 'mental_health_interview', 'phys_health_interview',
    'mental_vs_physical', 'obs_consequence'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Apply one-hot encoding
        # Note: drop_first=True must match your training phase
        input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_COLS, drop_first=True)

        # Reindex: This adds missing columns as 0 and removes extra columns
        # This ensures the input matches the 57 columns the model expects
        final_input = input_encoded.reindex(columns=MODEL_COLUMNS, fill_value=0)

        # Make prediction
        prediction_encoded = model.predict(final_input)

        # Decode the prediction (e.g., 0/1 back to 'No'/'Yes')
        prediction_label = le.inverse_transform(prediction_encoded)

        return jsonify({
            'status': 'success',
            'prediction': str(prediction_label[0])
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/', methods=['GET'])
def index():
    return "Mental Health Prediction API is running."

if __name__ == '__main__':
    # host='0.0.0.0' allows connections from outside the local machine
    app.run(debug=True, host='0.0.0.0', port=5000)
