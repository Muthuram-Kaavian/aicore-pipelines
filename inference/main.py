from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json["input"]
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        return jsonify({"predicted_price": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
