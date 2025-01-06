import requests
import json

url = "http://127.0.0.1:5000/predict"

data = {
    "features": [0.5, -1.2, 3.1, 0.8, 1.1, -0.3, 0.2, 1.9, -2.3, 0.1, 0.4, -0.7, 1.3, -0.2, 2.5, -1.4, 0.7, 2.8, -1.1, -0.5]
}

response = requests.post(url, json=data)

# Check if the response is empty or not JSON
if response.status_code == 200:
    try:
        response_json = response.json()
        prediction = response_json.get('prediction', None)
        if prediction is not None:
            print(f"Prediction: {prediction}")
        else:
            print("Prediction key not found in response.")
    except requests.exceptions.JSONDecodeError:
        print("Response is not in JSON format.")
        print("Raw response:", response.text)
else:
    print(f"Error: {response.status_code}, {response.text}")
