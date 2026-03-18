import pandas as pd
import joblib
from flask import Flask, request, jsonify

# ------------------ INITIALIZE APP ------------------
app = Flask(__name__)

# ------------------ LOAD MODEL ------------------
MODEL_PATH = "random_forest_model.joblib"
COLUMNS_PATH = "columns.pkl"

model = joblib.load(MODEL_PATH)
reference_columns = joblib.load(COLUMNS_PATH)

print(f"✅ Model loaded from {MODEL_PATH}")
print(f"✅ Columns loaded from {COLUMNS_PATH}")


# ------------------ PREPROCESS FUNCTION ------------------
def preprocess_input(data, reference_columns):
    # Convert input to DataFrame
    df_input = pd.DataFrame([data])

    # Handle gender mapping (same as training)
    if 'gender' in df_input.columns:
        df_input['gender'] = df_input['gender'].replace({
            'Male': 'Male',
            'Female': 'Female',
            'Non-binary': 'Non-binary',
            'Other': 'Other / Prefer not to say'
        })

    # Identify categorical columns
    categorical_cols = df_input.select_dtypes(include='object').columns.tolist()

    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

    # Add missing columns
    for col in reference_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Ensure correct order
    df_encoded = df_encoded[reference_columns]

    return df_encoded


# ------------------ PREDICTION ROUTE ------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # Preprocess input
        processed_data = preprocess_input(input_data, reference_columns)

        # Model prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0].max()

        # Optional: Convert prediction to readable output
        result = "High Risk" if prediction == 1 else "Low Risk"

        return jsonify({
            "prediction": int(prediction),
            "risk_level": result,
            "confidence": round(float(probability), 3),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500


# ------------------ HEALTH CHECK ROUTE ------------------
@app.route('/')
def home():
    return "🚀 Random Forest Mental Health API is running!"


# ------------------ MAIN ------------------
if __name__ == "__main__":
    print("🚀 Starting Random Forest API...")
    app.run(host="0.0.0.0", port=5000, debug=True)
