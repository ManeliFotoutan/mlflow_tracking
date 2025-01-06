from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import numpy as np

app = Flask(__name__)

model_name = "RandomForestModel"
model_version = "1"  
model_uri = f"models:/{model_name}/{model_version}"

model = mlflow.sklearn.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_data = np.array(data['features']).reshape(1, -1)  
        
        prediction = model.predict(input_data)
        
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)  

    