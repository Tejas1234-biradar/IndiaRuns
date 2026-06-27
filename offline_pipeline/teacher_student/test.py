import requests
import os

api_key = "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"
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(response.json())