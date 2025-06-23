import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
model = None

def init():
    global model
    model_path = "model.pkl"  # Required by SAP AI Core
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(" Model loaded")

@app.route("/v2/predict", methods=["POST"])
def predict():
    global model
    data = request.get_json()
    try:
        features = [
            data["MedInc"], data["HouseAge"], data["AveRooms"],
            data["AveBedrms"], data["Population"], data["AveOccup"],
            data["Latitude"], data["Longitude"]
        ]
        pred = model.predict([features])
        return jsonify({"prediction": float(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/v2/greet", methods=["GET"])
def greet():
    return "Hello! Model is ready." if model else "Model not loaded."

if __name__ == "__main__":
    print("🚀 Starting Flask app")
    init()
    app.run(host="0.0.0.0", port=9001)
