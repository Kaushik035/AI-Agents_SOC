import openai
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

# Dummy API key since real key is not used with the proxy
openai.api_key = "dummy"

# Load your proxy URL from environment
PROXY_URL = os.getenv("PROXY_URL")
if not PROXY_URL:
    raise ValueError("PROXY_URL is not set in the .env file")

def custom_chat_create(**kwargs):
    response = requests.post(PROXY_URL, json=kwargs)
    if response.status_code != 200:
        raise Exception(response.json())
    return response.json()

# Override the OpenAI ChatCompletion.create method
openai.ChatCompletion.create = custom_chat_create
