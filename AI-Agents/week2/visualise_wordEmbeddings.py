from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer('all-MiniLM-L6-v2')
# sentences = ["I love Apple products.", "An apple a day keeps the doctor away."]
sentences = [
    "He’s a real genius — failed every subject!",    # Sarcastic
    "She is extremely intelligent and successful."   # Literal
]

embeddings = embedder.encode(sentences)
cos_sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
print(f"Cosine Similarity: {cos_sim}")