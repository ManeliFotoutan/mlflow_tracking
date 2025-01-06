import requests
import json

url = "http://127.0.0.1:5000/predict"

data = {
    "features": [0.5, -1.2, 3.1, 0.8, 1.1, -0.3, 0.2, 1.9, -2.3, 0.1, 0.4, -0.7, 1.3, -0.2, 2.5, -1.4, 0.7, 2.8, -1.1, -0.5]
}

response = requests.post(url, json=data)

if response.status_code == 200:
    prediction = response.json()['prediction']
    print(f"Prediction: {prediction}")
else:
    print(f"Error: {response.status_code}, {response.json()}")
