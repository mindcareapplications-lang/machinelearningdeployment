import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ------------------ INITIALIZE APP ------------------
app = Flask(__name__)
CORS(app)

# ------------------ GLOBAL VARIABLES ------------------
MODEL = None
SCALER = None
COLUMNS = None

NUMERICAL_COLS = ['survey_year', 'age']

# ------------------ LOAD ASSETS ------------------
def load_model_assets():
    global MODEL, SCALER, COLUMNS
    try:
        print("📂 Files in directory:", os.listdir())

        MODEL = joblib.load("random_forest_model.joblib")
        SCALER = joblib.load("scaler.joblib")
        COLUMNS = joblib.load("columns.pkl")

        print("✅ Model, Scaler, Columns loaded successfully")

    except Exception as e:
        print(f"❌ Error loading assets: {e}")
        MODEL, SCALER, COLUMNS = None, None, None


# Load assets at startup
load_model_assets()


# ------------------ PREPROCESS FUNCTION ------------------
def preprocess_input(data_dict):
    input_df = pd.DataFrame([data_dict])

    # Handle gender mapping
    if 'gender' in input_df.columns:
        input_df['gender'] = input_df['gender'].replace({
            'Other': 'Other / Prefer not to say'
        })

    # ------------------ SCALING ------------------
    if SCALER:
        missing_cols = [col for col in NUMERICAL_COLS if col not in input_df.columns]
        if missing_cols:
            raise ValueError(f"Missing numerical columns: {missing_cols}")

        input_df[NUMERICAL_COLS] = SCALER.transform(input_df[NUMERICAL_COLS])

    # ------------------ ONE HOT ENCODING ------------------
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # ------------------ ALIGN COLUMNS ------------------
    input_final = input_encoded.reindex(columns=COLUMNS, fill_value=0)

    return input_final


# ------------------ ROUTES ------------------

@app.route('/')
def home():
    status = "Online" if MODEL else "Degraded"
    return jsonify({
        "service": "MindForge Prediction API",
        "status": status,
        "message": "POST data to /predict"
    })


@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded properly"
        }), 500

    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "No input data"}), 400

        # Preprocess
        processed = preprocess_input(data)

        # Predict
        proba = MODEL.predict_proba(processed)[0][1]
        prediction = "High Risk" if proba > 0.5 else "Low Risk"

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "probability": round(float(proba), 4)
        })

    except Exception as e:
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


# ------------------ START SERVER ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
