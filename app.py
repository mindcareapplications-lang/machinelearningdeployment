import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ------------------ INITIALIZE APP ------------------
app = Flask(__name__)
CORS(app) # Allows your frontend/PC app to talk to this API

# ------------------ LOAD ASSETS ------------------
# We load these once when the server starts
try:
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    reference_columns = joblib.load("columns.pkl")
    print("✅ All assets (Model, Scaler, Columns) loaded successfully.")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    model = scaler = reference_columns = None

# Define the numerical columns used during training
NUMERICAL_COLS = ['survey_year', 'age']

# ------------------ PREPROCESS FUNCTION ------------------
def preprocess_input(data_dict):
    """
    Matches the exact preprocessing steps from your Colab notebook:
    1. Standardization (Gender)
    2. Scaling (Numerical)
    3. One-Hot Encoding
    4. Column Alignment
    """
    # Create DataFrame from single input
    input_df = pd.DataFrame([data_dict])

    # 1. Standardize Gender (Matches your Colab replace logic)
    if 'gender' in input_df.columns:
        input_df['gender'] = input_df['gender'].replace({
            'Other': 'Other / Prefer not to say'
        })

    # 2. Scaling
    # We use the loaded scaler from training
    input_df[NUMERICAL_COLS] = scaler.transform(input_df[NUMERICAL_COLS])

    # 3. One-Hot Encoding
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # 4. Column Alignment
    # This ensures the input matches the 100+ columns the model expects
    # It adds missing columns as 0 and removes extra columns
    input_final = input_encoded.reindex(columns=reference_columns, fill_value=0)

    return input_final

# ------------------ ROUTES ------------------

@app.route('/')
def health_check():
    status = "Ready" if model else "Error (Check Logs)"
    return jsonify({"service": "MindForge Prediction API", "status": status})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model files missing on server"}), 500

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Preprocess and Predict
        processed_data = preprocess_input(data)
        
        # Get class probabilities
        prediction_proba = model.predict_proba(processed_data)[0]
        # Assuming index 1 is 'Yes' for treatment (treatment_Yes)
        probability = float(prediction_proba[1])
        prediction_label = "Yes" if probability > 0.5 else "No"

        return jsonify({
            "status": "success",
            "prediction": prediction_label,
            "probability": round(probability, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

# ------------------ START SERVER ------------------
if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    # host="0.0.0.0" is required for cloud accessibility
    app.run(host="0.0.0.0", port=port)
