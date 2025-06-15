import os
import fitz  # PyMuPDF
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from shared.newOpenAI import openai
from chromadb import PersistentClient

# 1. Load and chunk multiple PDF files by paragraph
def load_and_chunk_pdfs(file_paths):
    all_chunks = []
    for path in file_paths:
        doc = fitz.open(path)  # Open the PDF file
        for page_num in range(len(doc)):
            page_text = doc.load_page(page_num).get_text().strip()
            # Split by double newlines (which usually separate paragraphs)
            paragraphs = page_text.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if len(para) > 20:  # Filter out short or empty paragraphs
                    all_chunks.append({
                        "content": para,
                        "metadata": {
                            "source": os.path.basename(path),
                            "page": page_num + 1
                        }
                    })
    return all_chunks


# 2. Generate embeddings using SentenceTransformer
def generate_embeddings(chunks):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["content"] for chunk in chunks] # Extract text content from chunks, its a list of strings
    embeddings = embedder.encode(texts, convert_to_numpy=True) #numpy array
    return embeddings, embedder, texts, [chunk["metadata"] for chunk in chunks]

# 3. Create and store vector index using ChromaDB
def create_chroma_index(texts, embeddings, metadatas, persist_directory="chroma_store"):

    chroma_client = PersistentClient(path=persist_directory)

    # Check if the collection already exists and delete it if necessary 
    existing_collections = [col.name for col in chroma_client.list_collections()]   
    if "pdf_index" in existing_collections:
        chroma_client.delete_collection("pdf_index")

    collection = chroma_client.create_collection("pdf_index")

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=[f"chunk_{i}" for i in range(len(texts))]
    )
    return collection

""" 
documents: A list of plain text strings.

embeddings: A list of vectors (one per document).

metadatas: A list of dictionaries, where each dictionary is metadata for the corresponding document.

ids: Unique IDs per document.

📌 Chroma automatically matches these by index:

documents[0], embeddings[0], metadatas[0], and ids[0] are all considered part of the same record
 """



"""
ChromaDB returns query results in a list-of-lists format,
where each inner list corresponds to one query.

Even if we provide only a single query, the structure is still:

result = {
    'documents': [["chunk1 text", "chunk2 text", "chunk3 text"]],
    'metadatas': [[{"source": "file1.pdf", "page": 1}, {"source": "file1.pdf", "page": 2}, {"source": "file2.pdf", "page": 1}]],
    ...
}

To access the actual list of results for our only query,
we use [0] to extract the first (and only) inner list:

retrieved_chunks = result['documents'][0]
sources = result['metadatas'][0]
"""

# 4. Retrieve top-k similar chunks
def retrieve_chunks(question, embedder, collection, k=3):
    query_embedding = embedder.encode([question], convert_to_numpy=True).tolist()
    result = collection.query(query_embeddings=query_embedding, n_results=k)
    
    retrieved_chunks = result['documents'][0]
    sources = result['metadatas'][0]
    # print(f"Retrieved {len(retrieved_chunks)} relevant chunks for the question.")
    # print("Chunks:", retrieved_chunks)
    # print("Sources:", sources)
    
    return retrieved_chunks, sources

# 5. Construct prompt using context
def create_prompt(chunks, sources, question):
    context = "\n\n".join(chunks)
    source_text = "; ".join([f"{src['source']} (page {src['page']})" for src in sources])
    
    prompt = f"""
    You are a helpful assistant. Answer the question using only the notes below.
    If the answer is not found in the notes, reply: "I cannot find the answer in the provided notes."

    Notes:
    {context}

    Question: {question}
    Answer (include sources at the end if possible):
    """
    return prompt, source_text

# 6. Main chatbot loop
def run_chatbot():
    pdf_files = [
        "studyBuddy/week2/Assignment2.2/pdfs/pdf1.pdf",
        "studyBuddy/week2/Assignment2.2/pdfs/pdf2.pdf",
        "studyBuddy/week2/Assignment2.2/pdfs/pdf3.pdf"
    ]
    
    print("⏳ Loading and embedding PDF files...")
    chunks = load_and_chunk_pdfs(pdf_files)
    embeddings, embedder, texts, metadatas = generate_embeddings(chunks)
    collection = create_chroma_index(texts, embeddings, metadatas)

    print("✅ Chatbot ready. Type your question or 'quit' to exit.")
    
    while True:
        question = input("\nYou: ")
        if question.lower().strip() == "quit":
            print("Goodbye!")
            break

        relevant_chunks, sources = retrieve_chunks(question, embedder, collection)
        prompt, source_info = create_prompt(relevant_chunks, sources, question)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response["choices"][0]["message"]["content"]
        print(f"\nAssistant:\n{answer}\n\n📚 Source(s): {source_info}")

if __name__ == "__main__":
    run_chatbot()
