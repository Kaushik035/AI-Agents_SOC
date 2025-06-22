from multiprocessing import context
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tools import search_tavily, search_wikipedia, calculate
import os



# Short term memory for conversation history
conversation_history = []

def add_to_history(role, content):
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > 10:
        conversation_history.pop(0)

def history_of_conversation():
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    return history_text



# RAG Core Functionality
def load_and_chunk_document(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    if not text.strip():
        raise ValueError("The document is empty or contains only whitespace.")

    chunks = text.split('\n\n')
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 10]
    return chunks

def generate_embeddings(chunks):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return embeddings, embedder

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_chunks(question, embedder, index, chunks, k=2):
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(question_embedding, k)
    return [chunks[idx] for idx in indices[0]]


def generate_rag_response(question, relevant_chunks, generator):
    # Combine chat history and relevant document chunks
    history_text = history_of_conversation()
    context = "\n".join(relevant_chunks)
    # print("history_text:", history_text)
    
    # Final prompt
    prompt = f"{history_text}\nUser: {question}\n\nContext:\n{context}\n\nAssistant:"
    # Generate the response
    response = generator( prompt,
        max_new_tokens=256,
        num_return_sequences=1,
        truncation=True)
    generated_text = response[0]["generated_text"]

    # Parse answer
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[1].strip()
    else:
        answer = generated_text.strip()

    # Clean up formatting
    if not answer.endswith("."):
        last_period = answer.rfind(".")
        if last_period != -1:
            answer = answer[:last_period + 1]

    return answer


# Tool Router
def select_tool(query):
    query_lower = query.lower()
    if "search" in query_lower or "latest" in query_lower:
        return "tavily"
    elif "wiki" in query_lower or "who is" in query_lower:
        return "wikipedia"
    elif any(op in query_lower for op in ["+", "-", "*", "/", "calculate"]):
        return "calculator"
    else:
        return "rag"

def handle_query(query, generator, embedder, index, chunks):
    tool = select_tool(query)
    if tool == "tavily":
        result = search_tavily(query)
    elif tool == "wikipedia":
        result = search_wikipedia(query)
    elif tool == "calculator":
        expression = query.lower().split("calculate", 1)[1].strip() if "calculate" in query.lower() else query
        result = calculate(expression)
    else:
        relevant_chunks = retrieve_chunks(query, embedder, index, chunks)
        result = generate_rag_response(query, relevant_chunks, generator)

    add_to_history("user", query)
    add_to_history("assistant", result)
    return result

# Main
def run_chatbot(file_path):
    print("Loading document and models...")
    chunks = load_and_chunk_document(file_path)
    embeddings, embedder = generate_embeddings(chunks)
    index = create_faiss_index(embeddings)
    generator = pipeline("text-generation", model="gpt2")

    print("Welcome to the Study Buddy! Type 'quit' to exit.")
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break
        response = handle_query(query, generator, embedder, index, chunks)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    run_chatbot("AI-Agents/week3/notes.md")
