import os
print("📂 Flask working directory:", os.getcwd())
print("📦 Loading model from:", os.path.abspath('best_rf_model.pkl'))


from flask import Flask, request, jsonify
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load your trained Random Forest model
with open('best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "🚀 Hepatitis C Random Forest API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        # Expected format: {"features": [val1, val2, ...]}
        features = np.array(data['features']).reshape(1, -1)
        # Make prediction
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
