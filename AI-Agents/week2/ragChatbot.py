from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_and_chunk_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Split by paragraphs (double newlines), as generally paragraphs are separated by a line break
    chunks = text.split('\n\n')
    # Clean and filter chunks
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 10]
    return chunks

def generate_embeddings(chunks):
    # Load the embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Generate embeddings (returns a list of numpy arrays)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return embeddings , embedder

def create_faiss_index(embeddings):
    # Get embedding dimension, as we to tell faiss that calculate L2 distance of vectors of this dimension
    dimension = embeddings.shape[1]
    # Create a flat index with L2 distance
    index = faiss.IndexFlatL2(dimension)
    # Add embeddings to the index
    index.add(embeddings)
    return index

# Storage: We’ll keep the original chunks in a list to map back to the text after retrieval.

def retrieve_chunks(question, embedder, index, chunks, k=2):
    # Convert question to embedding
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    # Search for top-k similar chunks
    distances, indices = index.search(question_embedding, k)
    # Get the corresponding text chunks
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    return relevant_chunks

"""
distances: how far the matches are (smaller = more similar)

indices = [[12, 47]]
 “The chunks at position 12 and 47 are most similar to the question.” 

 indices[0]: Gets the list of matched indices (e.g., [12, 47]).

It’s [0] because index.search always returns a 2D array (even for one query).
 
 """


def generate_rag_response(question, relevant_chunks, generator):
    # Create the prompt
    context = "\n".join(relevant_chunks)
    prompt = f"Based on the following notes:\n{context}\n\nQuestion: {question}\nAnswer:"
    # Generate response
    response = generator(prompt, max_length=100, num_return_sequences=1)
    generated_text = response[0]["generated_text"]
    # Extract the answer
    answer = generated_text.split("Answer:")[1].strip()
    if not answer.endswith("."):
        last_period = answer.rfind(".")
        if last_period != -1:
            answer = answer[:last_period + 1]
    return answer


def run_rag_chatbot(file_path):
    # Initialize components
    print("Loading document and models...")
    chunks = load_and_chunk_document(file_path)
    embeddings, embedder = generate_embeddings(chunks)
    index = create_faiss_index(embeddings)
    generator = pipeline("text-generation", model="gpt2")
    
    print("Welcome to the RAG Study Buddy! Type 'quit' to exit.")
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            print("Goodbye!")
            break
        relevant_chunks = retrieve_chunks(question, embedder, index, chunks)
        answer = generate_rag_response(question, relevant_chunks, generator)
        print(f"Assistant: {answer}")

# Start the chatbot
run_rag_chatbot("AI-Agents/week2/notes.md")