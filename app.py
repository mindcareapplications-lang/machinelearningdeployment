import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Added for better connectivity
from sklearn.preprocessing import LabelEncoder
import os
import sys

# 1. INITIALIZE THE APP FIRST
app = Flask(__name__)
CORS(app) # This prevents connection reset errors from browsers

# 2. DEFINE PATHS
MODEL_PATH = 'random_forest_model.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib'

# 3. ROBUST LOADING (Runs once when the server starts)
try:
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found. Exiting application.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {e}. Exiting application.")
    sys.exit(1)

try:
    le = joblib.load(LABEL_ENCODER_PATH)
    print(f"Successfully loaded LabelEncoder from {LABEL_ENCODER_PATH}")
except FileNotFoundError:
    print(f"Error: LabelEncoder file '{LABEL_ENCODER_PATH}' not found. Exiting application.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading LabelEncoder: {e}. Exiting application.")
    sys.exit(1)

# 4. HARDCODED COLUMN DATA (Ensure this matches your training exactly)
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

original_categorical_cols = [
    'gender', 'country', 'self_employed', 'family_history', 'work_interfere',
    'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
    'wellness_program', 'seek_help', 'anonymity', 'leave',
    'mental_health_consequence', 'phys_health_consequence', 'coworkers',
    'supervisor', 'mental_health_interview', 'phys_health_interview',
    'mental_vs_physical', 'obs_consequence'
]

# 5. ROUTES
@app.route('/')
def home():
    # If you have an index.html in the templates folder, use: return render_template('index.html')
    return "<h1>MindCare ML API is Live!</h1><p>Send a POST request to /predict to get results.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])

        # One-hot encoding for the single input row
        input_encoded = pd.get_dummies(input_df, columns=original_categorical_cols, drop_first=True)

        # Align with the training columns
        final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Make prediction
        prediction_encoded = model.predict(final_input)
        prediction_label = le.inverse_transform(prediction_encoded)

        return jsonify({
            'status': 'success',
            'prediction': str(prediction_label[0])
        })

    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 400

# 6. RENDER DEPLOYMENT SETTINGS
if __name__ == '__main__':
    # Use the port Render assigns, or default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    # host='0.0.0.0' is required for cloud accessibility
    app.run(host='0.0.0.0', port=port)
