from shared.newOpenAI import openai
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_and_chunk_document(file_path = "studyBuddy/week2/Assignment2.1/my_note.md"):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    if not text.strip():
        raise ValueError("The document is empty or contains only whitespace.")

    # Split by paragraphs (double newlines), as generally paragraphs are separated by a line break
    chunks = text.split('\n\n') # This will split the text into chunks based on paragraphs,its a list of strings

    # Clean and filter chunks
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 10] # Filter out chunks that are too short, it's a list of strings
    return chunks

def generate_embeddings(chunks):
    # Load the embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Generate embeddings (returns a list of numpy arrays)
    embeddings = embedder.encode(chunks, convert_to_numpy=True) # the chunks list is converted to its respective embeddings, which is a numpy array of shape (n_chunks, embedding_dimension)
    return embeddings , embedder

def create_faiss_index(embeddings):
    # Get embedding dimension, as we to tell faiss that calculate L2 distance of vectors of this dimension
    dimension = embeddings.shape[1]
    # Create a flat index with L2 distance
    index = faiss.IndexFlatL2(dimension)
    # Add embeddings to the index
    index.add(embeddings)
    return index

def retrieve_chunks(question, embedder, index, chunks, k=2):
    # Convert question to embedding
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    # Search for top-k similar chunks
    distances, indices = index.search(question_embedding, k)
    # Get the corresponding text chunks
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    # print("Distances:", distances)
    # print(f"Retrieved {len(relevant_chunks)} relevant chunks for the question.")
    # print(relevant_chunks)
    return relevant_chunks

def create_prompt(relevant_chunks, question):
    # Join the relevant chunks into a single string
    relevant_content = "\n\n".join(relevant_chunks)
    prompt = f"""
    You are an AI Study Assistant. Answer the following question based ONLY on the provided notes.
    If the answer is directly present or reasonably inferred from the notes, use them.
    If there's no relevant information, say: 'I cannot find the answer in the provided notes.

    Notes:
    {relevant_content}

    Question: {question}
    Answer:
    """
    return prompt

def run_chatbot():
    # Load and chunk the document
    chunks = load_and_chunk_document()

    # Generate embeddings for the chunks
    embeddings, embedder = generate_embeddings(chunks)

    # Create a FAISS index for the embeddings
    index = create_faiss_index(embeddings)

    while True:
        question = input("\nEnter your question (or type 'quit' to exit): ")
        if question.lower() == "quit":
            print("Goodbye!")
            break

        # Retrieve relevant chunks based on the question
        relevant_chunks = retrieve_chunks(question, embedder, index, chunks)

        # Create the prompt using the relevant chunks
        prompt = create_prompt(relevant_chunks, question)

        # Get the response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response["choices"][0]["message"]["content"]
        print("\nAnswer:", answer)

if __name__ == "__main__":
    run_chatbot()
