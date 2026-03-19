import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Recommended for frontend-backend connection

# ------------------ INITIALIZE APP ------------------
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# ------------------ UTILITY: LOAD ASSETS ------------------
def load_assets():
    """Safely loads model, scaler, and columns with error handling."""
    try:
        model = joblib.load("random_forest_model.joblib")
        scaler = joblib.load("scaler.joblib")
        columns = joblib.load("columns.pkl")
        return model, scaler, columns
    except FileNotFoundError as e:
        print(f"❌ Error: Required file not found: {e.filename}")
        return None, None, None
    except Exception as e:
        print(f"❌ Unexpected loading error: {e}")
        return None, None, None

model, loaded_scaler, reference_feature_columns = load_assets()

# ------------------ CONFIGURATION ------------------
NUMERICAL_COLS = ['survey_year', 'age']

# ------------------ PREPROCESS FUNCTION ------------------
def preprocess_input(data_dict, reference_columns):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data_dict])

    # Standardize 'gender' if present
    if 'gender' in input_df.columns:
        input_df['gender'] = input_df['gender'].replace({'Other': 'Other / Prefer not to say'})

    # 1. Scaling
    for col in NUMERICAL_COLS:
        if col not in input_df.columns:
            raise ValueError(f"Required numerical column '{col}' is missing.")
    
    input_df[NUMERICAL_COLS] = loaded_scaler.transform(input_df[NUMERICAL_COLS])

    # 2. One-Hot Encoding
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # 3. Alignment (Vectorized approach)
    # Reindex fills missing columns with 0 and drops any extra columns in one step
    input_encoded = input_encoded.reindex(columns=reference_columns, fill_value=0)

    return input_encoded

# ------------------ ROUTES ------------------
@app.route('/', methods=['GET'])
def home():
    status = "Online" if model else "Offline (Files Missing)"
    return jsonify({"service": "Mental Health Prediction API", "status": status})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not initialized. Check server logs."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Run Preprocessing
        processed_data = preprocess_input(data, reference_feature_columns)

        # Generate Prediction
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        confidence = float(probabilities.max())

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "confidence": round(confidence, 4),
            "class_probabilities": probabilities.tolist()
        })

    except ValueError as ve:
        return jsonify({"error": str(ve), "status": "validation_failed"}), 400
    except Exception as e:
        return jsonify({"error": "Internal processing error", "details": str(e)}), 500

# ------------------ EXECUTION ------------------
if __name__ == "__main__":
    # Use the PORT environment variable provided by Render
    port = int(os.environ.get("PORT", 5000))
    # host='0.0.0.0' is required for cloud accessibility
    app.run(host="0.0.0.0", port=port)
