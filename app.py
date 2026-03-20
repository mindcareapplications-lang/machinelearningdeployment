import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ------------------ INITIALIZE APP ------------------
app = Flask(__name__)
CORS(app)  # Enables cross-origin requests for your frontend

# ------------------ LOAD ASSETS ------------------
def load_model_assets():
    """Loads all necessary machine learning files globally."""
    try:
        # These filenames must match exactly what you uploaded to GitHub
        model = joblib.load("random_forest_model.joblib")
        scaler = joblib.load("scaler.joblib")
        columns = joblib.load("columns.pkl")
        print("✅ Success: Model, Scaler, and Columns loaded.")
        return model, scaler, columns
    except Exception as e:
        print(f"❌ Error loading assets: {e}")
        return None, None, None

# Load assets once when the server starts
loaded_model, loaded_scaler, reference_feature_columns = load_model_assets()

# Constants from training
NUMERICAL_COLS = ['survey_year', 'age']

# ------------------ PREPROCESS FUNCTION ------------------
def preprocess_input(data_dict, reference_columns, scaler):
    """
    Transforms raw JSON input into the exact format required by the model.
    """
    # 1. Convert to DataFrame
    input_df = pd.DataFrame([data_dict])

    # 2. Standardize Gender (Matches your Colab notebook logic)
    if 'gender' in input_df.columns:
        input_df['gender'] = input_df['gender'].replace({
            'Other': 'Other / Prefer not to say'
        })

    # 3. Apply Scaling
    if all(col in input_df.columns for col in NUMERICAL_COLS):
        input_df[NUMERICAL_COLS] = scaler.transform(input_df[NUMERICAL_COLS])

    # 4. One-Hot Encoding
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # 5. Align Columns (Crucial Step)
    # This adds missing columns as 0 and removes extra ones to match training data
    input_final = input_encoded.reindex(columns=reference_columns, fill_value=0)

    return input_final

# ------------------ API ROUTES ------------------

@app.route('/')
def home():
    """Health check route to verify the service is live."""
    status = "Online" if loaded_model else "Degraded (Assets Missing)"
    return jsonify({
        "service": "MindForge Prediction API",
        "status": status,
        "message": "Send a POST request to /predict"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handles incoming prediction requests."""
    if not loaded_model:
        return jsonify({"error": "Model files not found on server"}), 500

    try:
        # Parse incoming JSON
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Preprocess the data
        processed_data = preprocess_input(data, reference_feature_columns, loaded_scaler)

        # Generate Prediction
        # [:, 1] gets the probability for the 'Yes' class (treatment needed)
        prediction_proba = loaded_model.predict_proba(processed_data)[:, 1][0]
        prediction_label = "Yes" if prediction_proba > 0.5 else "No"

        return jsonify({
            "status": "success",
            "prediction": prediction_label,
            "probability": round(float(prediction_proba), 4)
        })

    except Exception as e:
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500

# ------------------ START SERVER ------------------
if __name__ == "__main__":
    # Render assigns a port dynamically via environment variables
    port = int(os.environ.get("PORT", 5000))
    
    # Use 0.0.0.0 to allow external traffic in cloud environments
    app.run(host="0.0.0.0", port=port)
