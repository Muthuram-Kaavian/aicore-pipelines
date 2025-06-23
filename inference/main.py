from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved model
model = joblib.load("house_price_model.joblib")

# Feature order
FEATURE_ORDER = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                 "Population", "AveOccup", "Latitude", "Longitude"]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # JSON from Postman

        # Ensure all required features are present
        if not all(feature in data for feature in FEATURE_ORDER):
            return jsonify({"error": "Missing one or more required features."}), 400

        # Convert dict to array in correct order
        input_values = [data[feature] for feature in FEATURE_ORDER]
        input_array = np.array([input_values])  # 2D array

        prediction = model.predict(input_array).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
