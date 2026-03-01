import requests
import numpy as np

sequence = np.random.rand(30, 29).tolist()

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"sequence": sequence}
)

print(response.json())
