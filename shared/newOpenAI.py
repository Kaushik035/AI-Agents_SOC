# import openai
# import requests
# import os
# from dotenv import load_dotenv

# load_dotenv()  # Load .env file

# # Dummy API key since real key is not used with the proxy
# openai.api_key = "dummy"

# # Load your proxy URL from environment
# PROXY_URL = os.getenv("PROXY_URL")
# if not PROXY_URL:
#     raise ValueError("PROXY_URL is not set in the .env file")

# def custom_chat_create(**kwargs):
#     response = requests.post(PROXY_URL, json=kwargs)
#     if response.status_code != 200:
#         raise Exception(response.json())
#     return response.json()

# """ json=kwargs tells requests to:

# Convert kwargs to a JSON string.

# Set the Content-Type header to application/json """

# # Override the OpenAI ChatCompletion.create method
# openai.ChatCompletion.create = custom_chat_create


import os
from dotenv import load_dotenv
from openai import OpenAI
import openai as openai_namespace  # only for monkey-patching

load_dotenv()

# Initialize OpenAI SDK client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


# Monkey-patch to keep openai.ChatCompletion.create compatible
def custom_chat_create(**kwargs):
    return client.chat.completions.create(**kwargs)

# Inject patch
openai_namespace.ChatCompletion.create = custom_chat_create
