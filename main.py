from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Load the model and preprocessing tools
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Market Crash Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input data (JSON)
        data = request.get_json()
        features = pd.DataFrame([data])
        
        # Preprocess the input data
        features_imputed = pd.DataFrame(imputer.transform(features), columns=features.columns)
        features_scaled = pd.DataFrame(scaler.transform(features_imputed), columns=features.columns)
        
        # Make predictions
        probabilities = model.predict_proba(features_scaled)[:, 1]  # Probability of crash
        predictions = (probabilities >= 0.4).astype(int)  # Apply threshold
        
        # Respond with prediction
        return jsonify({
            "crash_probability": probabilities[0],
            "prediction": int(predictions[0])  # 1 = Crash, 0 = No Crash
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
