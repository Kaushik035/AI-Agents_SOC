import json
import os
import requests
from datetime import datetime
from shared.newOpenAI import openai
import dotenv
dotenv.load_dotenv()

API_URL = os.getenv("PROXY_URL")

HISTORY_FILE = "conversation_history.json"
conversation_history = []
entities = {}
MAX_TOKENS = 3000

def load_history():
    global conversation_history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            conversation_history = json.load(f)

def save_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump(conversation_history, f, indent=2)

def add_to_history(role, content):
    conversation_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    save_history()

def estimate_tokens(text):
    return len(text) // 4

def get_relevant_history(query, max_messages=5):
    query_words = set(query.lower().split())
    scored_messages = []
    for msg in conversation_history:
        msg_words = set(msg["content"].lower().split())
        overlap = len(query_words & msg_words)
        if overlap > 0:
            scored_messages.append((overlap, msg))
    scored_messages.sort(key=lambda x: x[0], reverse=True)
    return [msg for _, msg in scored_messages[:max_messages]]

def get_optimized_context(query):
    recent_msgs = conversation_history[-3:]
    relevant_msgs = get_relevant_history(query, max_messages=3)
    context_msgs = list({msg["content"]: msg for msg in recent_msgs + relevant_msgs}.values())
    total_tokens = 0
    final_context = []
    for msg in context_msgs:
        msg_tokens = estimate_tokens(msg["content"])
        if total_tokens + msg_tokens < MAX_TOKENS:
            final_context.append(msg)
            total_tokens += msg_tokens
    return final_context

def extract_entities(text):
    words = text.split()
    for word in words:
        if word.istitle() and len(word) > 2:
            entities[word] = {"context": text, "timestamp": datetime.now().isoformat()}

def get_entity_context(entity):
    return entities.get(entity, {}).get("context", "")

def summarize_context():
    full_context = " ".join([msg["content"] for msg in conversation_history])
    if estimate_tokens(full_context) < 50:
        return full_context
    summary_prompt = f"Summarize the following conversation concisely:\n{full_context}"
    summary_context = [{"role": "user", "content": summary_prompt}]
    return call_openai(summary_prompt, summary_context)

def call_openai(query, context):
    headers = {"Content-Type": "application/json"}
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in context]
    messages.append({"role": "user", "content": query})
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return "API error."

def run_chatbot():
    load_history()
    print("Welcome to the Study Buddy! Type 'quit' to exit.")
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break
        add_to_history("user", query)
        extract_entities(query)
        context = get_optimized_context(query)
        entity_context = get_entity_context(query.split()[0])
        if entity_context:
            context.append({"role": "system", "content": f"Note: User mentioned {query.split()[0]}: {entity_context}"})
        if len(conversation_history) > 10:
            summary = summarize_context()
            context.append({"role": "system", "content": f"Summary: {summary}"})
        response = call_openai(query, context)
        add_to_history("assistant", response)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    run_chatbot()