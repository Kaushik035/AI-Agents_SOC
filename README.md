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



## 📅 Week 4 Summary: State Management,Tool Chaining, Persona Engineering, Reasoning Framework

In **Week 4**, I modularized and upgraded the Study Buddy chatbot into a more intelligent assistant with **stateful memory**, **optimized context**, and **entity awareness** using `spaCy` and `tiktoken`, implemented **tool chaining**, **persona engineering**, **reasoning framework**



## 🧠 State Management Features Implemented

- **Persistent History**:
  - Conversation history is saved to and loaded from `conversation_history.json`.
  - Each message includes `role`, `content`, and `timestamp`.

- **Token-Aware Context Window** using `tiktoken`:
  - Dynamically selects messages that fit within a max token budget (e.g., 1200 tokens).
  - Prioritizes recent + relevant messages.

- **Relevant History Retrieval**:
  - Uses keyword overlap with current query to fetch past messages that match contextually.
  - Scores and returns top N overlaps.

- **Deduplication** of context:
  - Avoids repeated messages by de-duping based on content before LLM call.

- **Entity Tracking with `spaCy`**:
  - Named entities (like *Einstein*, *MRI*, *Quantum Tunneling*) are extracted on each user query.
  - Context around the entity is stored and recalled later if reused in a query.

- **Context Summarization**:
  - Once history grows beyond 10 turns, a concise summary is generated using OpenAI.
  - Injected as a `system` message in future prompts.



## 🛠 Tool Chaining (Multi-Step Tool Execution)

The system includes a **modular and resilient tool chaining pipeline**, designed to orchestrate multiple tools like Wikipedia, Tavily, and a calculator in a logical sequence. It supports input/output dependency resolution, conditional execution, and fallback handling — enabling a wide range of queries like:

> 🧠 _“What is 3 times the population of France?”_  
> 📈 _“Search the latest news on electric vehicles”_  
> ➕ _“Calculate 7 + 3?”_

We follow four key design principles:

---

### ✅ 1. Sequential Tool Execution

We define multi-step tool chains (e.g., `calc_with_lookup`) in a fixed order:
- First, extract numeric multiplier and subject (`France`, `population`)
- Query Wikipedia to get a base number (e.g., 67 million)
- Run calculation: `3 × 67 million`

This is implemented in:
```python
def run_tool_chain(query: str) -> str
```

---

### ✅ 2. Output Validation and Error Recovery

Each tool result (e.g., Wikipedia summary or calculator output) is **validated** before further use.

If the output is missing, ambiguous, or non-numeric:

- ❌ The chain halts
- 🧾 A clear error message is returned
- 🧠 If needed, it falls back to a direct OpenAI call using:

```python
def fallback_openai(query: str) -> str
```

### ✅ 3. Dependency Management Between Tools

In chained flows like `calc_with_lookup`:

- The output of Wikipedia (e.g., `"The population of France is 67 million"`)
- Is parsed using `extract_first_number()`
- And passed as input to the calculator

This ensures **clean data flow**, similar to a Unix-style pipeline:

```text
Wikipedia → Extract Number → Calculator → Final Result
```

### ✅ 4. Conditional Execution

The function `detect_intent(query)` determines the intent and triggers the correct tool:

| Sample Query                          | Tool Chain Triggered                      |
|--------------------------------------|-------------------------------------------|
| "Search the latest news on EVs"      | `tavily_search`                           |
| "GDP of Brazil times 5"              | `wikipedia → extract → calculate`         |
| "What is 11 + 7 / 2?"                | `calculator` only                         |
| "Who was Srinivasa Ramanujan?"       | `wikipedia` only                          |

This logic keeps the **orchestration clean, adaptive, and scalable**.


## Persona Engineering



### 🗣️ 1. Response Style Conditioning
- Adjusts tone based on inferred `user_level` (e.g., `high_school`, `grad_student`, `expert`).
- Uses soft rules and heuristics to simplify or enrich responses.
- Automatically adapts based on user queries and context.

```python
def infer_user_level(text: str) -> str
```

### 📚 2. Domain-Specific Language Modeling

This module adapts the assistant's language to match the **subject domain** of the user's query — ensuring correct vocabulary, terminology, and examples.

#### ✅ How It Works
- **Keyword Matching**: Checks for domain-specific terms (e.g., `algorithm`, `photosynthesis`, `equation`).
- **🧠 Embedding-Based Detection**: Uses `sentence-transformers` to compare the query to known domain labels (e.g., `physics`, `cs`, `history`) via cosine similarity.
- **Domain Prompts**: Each detected domain injects a custom system prompt that guides the assistant's tone and accuracy.

#### 🔍 Domain Detection Function

```python
def detect_domain(query: str) -> str:
  ```


### 🔒 3. Ethical Guardrails & Safety

This module ensures the assistant avoids harmful, biased, or offensive outputs by performing a layered compliance check.

#### ✅ How It Works
- **AI-Based Moderation**: Uses [Detoxify](https://github.com/unitaryai/detoxify) to scan text for:
  - Toxicity
  - Identity attacks
  - Severe insults
- **Fallback Checks**:
  - Matches hardcoded `SENSITIVE_TERMS`
  - Flags biased language like "superior race", "better than"
- **Logging**:
  - Every flagged response is appended to `guardrail_log.txt` for audit and analysis

#### 🔍 Guardrail Check Function

```python
def check_ethical_compliance(text: str) -> Tuple[bool, str]:
  ```


  ## Reasoning Framework 



## 🧩 Key Components

### 1. Plan-Execute-Refine (PER) Strategy

A three-stage reasoning pipeline:

- **Plan** – Generate a step-by-step plan for solving the query.
- **Execute** – Follow the plan to generate a detailed answer.
- **Refine** – Review and rewrite for clarity, correctness, and completeness.

> ✅ Use PER for procedural queries like *how*, *why*, *derive*, or *solve*.


### 2. Candidate Answer Generation

Each query runs through multiple answer generators:

- **RAG + Tool**: Combines retrieved notes and external tool output.
- **Plan-Execute-Refine**: For multi-step reasoning tasks.
- **Tool-only**: Uses tool outputs directly if available.
- **General-LLM**: Pure LLM fallback answer.


### 3. Confidence Scoring

Each answer is evaluated via a custom scoring function `_confidence()`:

```python
score = 0.5 * semantic_similarity(query, response) \
      + 0.3 * keyword_overlap(query, response) \
      + 0.2 * no_error_flags(response)

if origin in TOOL_DOMAINS:
    if len(response.split()) < 60: score += 0.15
    if overlap > 0.5: score += 0.15
```

### 📄 Detailed implementation and setup can be found in the [Week 4 README](./studyBuddy/week4/README.md).

---


