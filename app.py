import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Added for better connectivity
import os

# 1. Initialize the app FIRST
app = Flask(__name__)
CORS(app) # This helps if you're calling this API from a frontend/website

# 2. Load models and encoders
# Make sure these files are in your main project folder on GitHub
model = joblib.load('random_forest_model.joblib')
le = joblib.load('label_encoder.joblib')

# 3. Define Column Lists
model_columns = [
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
    'anonymity_Yes', 'anonymity_Not sure', 'leave_Don\'t know',
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

original_categorical_cols = [
    'gender', 'country', 'self_employed', 'family_history', 'work_interfere',
    'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
    'wellness_program', 'seek_help', 'anonymity', 'leave',
    'mental_health_consequence', 'phys_health_consequence', 'coworkers',
    'supervisor', 'mental_health_interview', 'phys_health_interview',
    'mental_vs_physical', 'obs_consequence'
]

# 4. Define Routes
@app.route('/')
def home():
    return {"status": "online", "message": "ML API is running"}, 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])

        # Apply one-hot encoding
        input_encoded = pd.get_dummies(input_df, columns=original_categorical_cols)

        # Reindex to match training columns
        final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Make prediction
        prediction_encoded = model.predict(final_input)

        # Inverse transform to 'Yes' or 'No'
        prediction_label = le.inverse_transform(prediction_encoded)

        return jsonify({'prediction': str(prediction_label[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# 5. Run logic (Render will use the PORT env variable)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
