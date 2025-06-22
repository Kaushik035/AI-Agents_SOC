from shared.newOpenAI import openai
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tools import search_tavily, search_wikipedia, calculate

#RAG Core Functionality
def load_and_chunk_document(file_path="studyBuddy/week3/notes/my_note.md"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    if not text.strip():
        raise ValueError("Document is empty.")
    chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 10]
    return chunks

def generate_embeddings(chunks):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return embeddings, embedder

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_chunks(question, embedder, index, chunks, k=2):
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(question_embedding, k)
    return [chunks[i] for i in indices[0]]

#short term memory for conversation history
conversation_history = []

def add_to_history(role, content):
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > 10:
        conversation_history.pop(0)

def previous_conversation_history():
    if not conversation_history:
        return "No previous conversation history."
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    # print("Previous conversation history:", history_text)
    return history_text

# tool selection based on query
def select_tool(query):
    q = query.lower()
    if "search" in q or "latest" in q:
        return "tavily"
    elif "wiki" in q or "who is" in q or "who was" in q:
        return "wikipedia"
    elif any(op in q for op in ["+", "-", "*", "/", "calculate"]):
        return "calculator"
    else:
        return "none"

#getting output from the selected tool
def get_tool_output(tool, query):
    if tool == "tavily":
        return search_tavily(query)
    elif tool == "wikipedia":
        return search_wikipedia(query)
    elif tool == "calculator":
        return calculate(query.lower().split("calculate", 1)[-1].strip() if "calculate" in query.lower() else query)
    return "No external tool used."

#prompt including relevant chunks and tool output and history of conversation
def create_prompt(relevant_chunks, tool_output, question):
    notes = "\n\n".join(relevant_chunks)
    return f"""
You are an intelligent AI Study Assistant.

Context from previous conversation:
{previous_conversation_history()}

Relevant notes:
{notes}

External tool output:
{tool_output}

Now answer the question clearly, or say "I cannot find the answer in the provided notes or tools" if nothing about the question is present in the provided context(Relevant Notes or External Tool Output).

User Question: {question}
Assistant:"""


def run_chatbot():
    chunks = load_and_chunk_document()
    embeddings, embedder = generate_embeddings(chunks)
    index = create_faiss_index(embeddings)

    print("Welcome to the Study Buddy! Type 'quit' to exit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break

        tool = select_tool(query)
        # print(f"Selected tool: {tool}")
        tool_output = get_tool_output(tool, query)
        # print(f"Tool output: {tool_output}")
        relevant_chunks = retrieve_chunks(query, embedder, index, chunks)
        prompt = create_prompt(relevant_chunks, tool_output, query)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response["choices"][0]["message"]["content"]
        print("\nAssistant:", answer)

        add_to_history("user", query)
        add_to_history("assistant", answer)


if __name__ == "__main__":
    run_chatbot()
