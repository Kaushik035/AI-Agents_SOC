# Steps for assignment 2.1

This project demonstrates a simple Retrieval-Augmented Generation (RAG) chatbot built using:
- `SentenceTransformers` to create text embeddings
- `FAISS` for fast similarity-based retrieval
- `OpenAI` for answering questions based only on retrieved chunks from study notes

The chatbot answers finance-related questions **based only on your provided notes** (`my_notes.md`) using a controlled prompt.

---
### 1. Install dependencies

```bash
 pip install -r requirements.txt
```
### 2. Create an .env file in the root directory and add your proxy URL to the .env file:

> PROXY_URL=https://socapi.deepaksilaych.me/student1


(Run the 3rd command only if you haven't used earlier, if you used earlier you should be able to see a folder named 'shared' in your root directory)
### 3. Install the shared package in editable mode:(This makes the shared/ module directly available to your script—no need for PYTHONPATH)

> pip install -e .

### 4. From the project root, run it:

> python studyBuddy/week2/Assignment2.1/rag_openAI_chatbot.py (can directly run using the Run button in vs code)


🧪 Testing
Here’s how to test:

✅ 1. Clearly in notes

How does behavioral bias affect investment decisions?

Response: Behavioral biases affect investment decisions by influencing investor behavior and market outcomes. Common biases such as overconfidence, loss aversion, herd behavior, and anchoring can lead to mispricing of assets, bubbles, or crashes. Understanding these tendencies helps financial professionals and investors make more informed and objective decisions.

✅ 2. Partially in notes

What are some ways to manage liquidity risk?

Response: Managing liquidity risk can be done using various tools, including insurance, hedging with derivatives, and maintaining reserves.

✅ 3. Not in notes

What are the key principles of Islamic finance?

Response: I cannot find the answer in the provided notes.

> Alter value of k, for getting more to k relevant chunk



