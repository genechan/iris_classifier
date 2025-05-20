from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_model.pkl')
species = ['setosa', 'versicolor', 'virginica']

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()
    features = [
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]
    # Convert to array for prediction
    features = np.array(features).reshape(1, -1)
    # Predict
    prediction = model.predict(features)[0]
    return jsonify({'species': species[prediction]})

@app.route('/', methods=['GET'])
def home():
    return "Iris Classifier API is running!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)