import requests
import json

with open("test_api.json") as f:
    data = json.load(f)

response = requests.post(url="http://127.0.0.1:1234/invocations", headers={"Content-Type": "application/json"}, json=data)

print(response.json())
