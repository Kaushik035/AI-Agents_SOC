## You must see two folders 

### 1. [AI-Agents](./AI-Agents/)

- It contains the week wise task that I learn.
- Here I experiment with the concepts, see logs to debug, how everything works behind the scene.

### 2. [studyBuddy](./studyBuddy/)

- It contains the week wise implementation of everything I learn.
- Here I apply the concepts that I learn with some modifications.
- It will be your interested folder, If you want to see the final results.


# 📘 Summary of Work Across Weeks in the [studyBuddy](./studyBuddy/) folder



## 📅 Week 1 Summary: Introduction to Generative AI
In Week 1, I explored the foundations of Generative AI and how to integrate it into applications.

### 🔍 Key Learnings:

- **What is Generative AI**  
  Learned that generative AI refers to models that can create new content like text, images, or music based on patterns learned from data.

- **How Generative AI Works**  
  Understood the role of neural networks (especially transformers) and how these models generate outputs using token prediction.

- **Benefits Over Traditional AI**  
  Discovered that generative models are more flexible and better at handling open-ended tasks compared to classification or rule-based AI.

- **Hands-on Integration Steps**  
  Practiced how to load a model (locally or via API), design input prompts, generate output, post-process responses, and build a basic interface for user interaction.

- **Tooling Options**  
  Got introduced to both local LLMs (like those on Hugging Face) and API-based ones (like OpenAI, Claude), along with pros/cons for each.

- **Prompt Engineering Basics**  
  Learned how to design effective prompts and structure queries to get better responses from LLMs.


### 📄 Detailed implementation and setup can be found in the [Week 1 README](./studyBuddy/week1/README.md).

---

## 📅 Week 2 Summary: Building a Retrieval-Augmented Generation (RAG) Pipeline
In Week 2, I implemented a full RAG pipeline to allow querying over personal notes and documents.

### 🛠️ Key Components Built:

- **RAG Pipeline:**
  - Loaded and chunked markdown notes.
  - Created embeddings using `all-MiniLM-L6-v2` model from Sentence Transformers.
  - Used **FAISS** to index and search over chunk embeddings based on similarity.
  - Retrieved relevant context to answer user queries using OpenAI’s `gpt-3.5-turbo`.

- **Console-Based Chatbot:**
  - Built a terminal chatbot that responds only using information from the notes.
  - If the answer was not found in the notes, it responded accordingly.

### 📚 Document Parsing:

- Parsed content from **3 PDF files** and extracted paragraphs as knowledge chunks.
- Implemented tracking of **source documents and page numbers** for each chunk.
- Ensured responses cited **which document** the information came from.

### 🔍 Additional Tools:

- Integrated **ChromaDB** (in parallel with FAISS) to explore document storage and semantic retrieval using persistent vector storage.

### 📄 Detailed implementation and setup can be found in the [Week 2 README](./studyBuddy/week2/README.md).

---

## 📅 Week 3 Summary: Enhancing the Study Buddy with Tools and Memory

In Week 3, I upgraded the basic RAG-based chatbot into a more powerful **AI Study Buddy** with tools and conversation memory.


- Added **tool integration** into the chatbot:
  - 🔍 `Tavily` for live web search.
  - 📚 `Wikipedia` API for encyclopedia-style summaries.
  - ➕ A safe Python calculator using `ast` module for math queries.
- Developed a **tool selector** to determine which tool (or RAG) to use based on the user's question.
- Enhanced the pipeline to **inject tool responses alongside retrieved notes** for richer LLM answers.
- Preserved conversation **memory** for multi-turn conversations.
- Added test examples to demonstrate:
  - Tool triggers (search, wiki, calculator).
  - Memory-aware follow-ups.
  - Pure RAG responses when notes are enough.

📁 Code files:
- `tool-memoryChatbot.py` → Main chatbot script with memory + tools + RAG.
- `tools.py` → Modular tool functions (Tavily, Wikipedia, Calculator).

### 📄 Detailed implementation and setup can be found in the [Week 3 README](./studyBuddy/week3/README.md).

---
