import time
import numpy as np
import requests

test_data = [
    [0.5, -1.2, 3.1, 0.8, 1.1, -0.3, 0.2, 1.9, -2.3, 0.1, 0.4, -0.7, 1.3, -0.2, 2.5, -1.4, 0.7, 2.8, -1.1, -0.5],
    [0.1, -0.5, 2.3, 1.2, 0.9, 0.2, -0.1, 1.4, -1.5, -0.1, 0.2, 0.3, 1.4, 0.5, -2.5, -1.0, 1.0, -0.2, 2.1, 0.6]
]

url = "http://127.0.0.1:5000/predict"

start_time = time.time()

for features in test_data:
    data = {"features": features}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        prediction = response.json()['prediction']
        print(f"Prediction: {prediction}")
    else:
        print(f"Error: {response.status_code}, {response.json()}")

end_time = time.time()
latency = end_time - start_time
print(f"Total latency for testing: {latency:.4f} seconds")
