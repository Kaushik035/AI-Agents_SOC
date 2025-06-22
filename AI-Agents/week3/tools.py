import requests
import wikipedia
import ast
import operator
from dotenv import load_dotenv
import os


load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Travily for searching web content
def search_tavily(query, api_key=TAVILY_API_KEY):
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": True
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get("answer", "No answer found.")
    return "Search failed."


# Wikipedia for searching general knowledge
def search_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple options found: {e.options}"
    except wikipedia.exceptions.PageError:
        return "No page found."


# Simple calculator using Python's ast module for safe evaluation
def calculate(expression):
    try:
        node = ast.parse(expression, mode='eval')
        result = eval_(node.body)
        return str(result)
    except Exception:
        return "Calculation error."

def eval_(node):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        left = eval_(node.left)
        right = eval_(node.right)
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
        return ops[type(node.op)](left, right)
    else:
        raise ValueError("Unsupported expression")