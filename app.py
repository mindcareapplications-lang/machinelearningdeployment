import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import os
import sys # Import sys for exiting the application
@app.route('/')
def home():
    return "<h1>MindCare ML API is Live!</h1><p>The model is ready. Send a POST request to /predict to get started.</p>"

app = Flask(__name__)

# Define paths to model and LabelEncoder files
MODEL_PATH = 'random_forest_model.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib'

# --- Robust Loading of Model and LabelEncoder ---
try:
    # Load the trained Random Forest Classifier model
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found. Exiting application.")
    sys.exit(1) # Exit if model is not found
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}. Exiting application.")
    sys.exit(1) # Exit for other loading errors

try:
    # Load the LabelEncoder object
    le = joblib.load(LABEL_ENCODER_PATH)
    print(f"Successfully loaded LabelEncoder from {LABEL_ENCODER_PATH}")
except FileNotFoundError:
    print(f"Error: LabelEncoder file '{LABEL_ENCODER_PATH}' not found. Exiting application.")
    sys.exit(1) # Exit if LabelEncoder is not found
except Exception as e:
    print(f"Error loading LabelEncoder from {LABEL_ENCODER_PATH}: {e}. Exiting application.")
    sys.exit(1) # Exit for other loading errors
# --- End Robust Loading ---


# Columns used in the X_encoded DataFrame during training (hardcoded for deployment)
# This list must exactly match the columns present during model training after one-hot encoding
model_columns = ['respondent_id', 'survey_year', 'age', 'gender_Male',
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
       'obs_consequence_Yes']

# Categorical columns from the original X DataFrame (before one-hot encoding, 'comments' already dropped)
original_categorical_cols = ['gender', 'country', 'self_employed', 'family_history', 'work_interfere',
                             'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
                             'wellness_program', 'seek_help', 'anonymity', 'leave',
                             'mental_health_consequence', 'phys_health_consequence', 'coworkers',
                             'supervisor', 'mental_health_interview', 'phys_health_interview',
                             'mental_vs_physical', 'obs_consequence']

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            input_df = pd.DataFrame([data])

            # Apply one-hot encoding to input data
            input_encoded = pd.get_dummies(input_df, columns=original_categorical_cols, drop_first=True)

            # Reindex to ensure all model columns are present and in the correct order
            # Fill missing columns with 0, as they were not present in the input for one-hot encoding
            final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

            # Make prediction
            prediction_encoded = model.predict(final_input)

            # Inverse transform to get original labels ('Yes' or 'No')
            prediction_label = le.inverse_transform(prediction_encoded)

            return jsonify({'prediction': prediction_label[0]})

        except Exception as e:
            return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Render provides a PORT environment variable; we MUST use it.
    port = int(os.environ.get("PORT", 5000))
    # Use 0.0.0.0 to let Render connect to the app
    app.run(host='0.0.0.0', port=port)
