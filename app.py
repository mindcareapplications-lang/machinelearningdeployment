import pandas as pd
import joblib
from flask import Flask, request, jsonify

# ------------------ INITIALIZE APP ------------------
app = Flask(__name__)

# ------------------ LOAD FILES ------------------
model = joblib.load("random_forest_model.joblib")
loaded_scaler = joblib.load("scaler.joblib")
reference_feature_columns = joblib.load("columns.pkl")

print("✅ Model, Scaler, and Columns loaded successfully")

# ------------------ DEFINE NUMERICAL COLUMNS ------------------
numerical_cols = ['survey_year', 'age']

# ------------------ PREPROCESS FUNCTION ------------------
def preprocess_input(data_dict, reference_columns):
    input_df = pd.DataFrame([data_dict])

    # Gender mapping
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
    for col in reference_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[reference_columns]

    return input_encoded


# ------------------ API ROUTE ------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        processed_data = preprocess_input(data, reference_feature_columns)

        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0].max()

        return jsonify({
            "prediction": int(prediction),
            "confidence": float(probability),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500


# ------------------ HOME ROUTE ------------------
@app.route('/')
def home():
    return "🚀 Mental Health Prediction API Running"


# ------------------ TEST (OPTIONAL) ------------------
if __name__ == "__main__":
    print("🚀 Starting Random Forest API...")
    app.run(host="0.0.0.0", port=5000, debug=True)
