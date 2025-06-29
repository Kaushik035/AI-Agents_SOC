# 🧠 Conversation History Management

## 📘 What Is Conversation History Management?

Conversation history management is the process of:
- **Storing all messages** between the user and the assistant
- **Persisting** them across sessions
- **Retrieving** relevant past messages dynamically, not just the most recent

### 🔍 Benefits:
- ✅ **Long-Term Memory**: Retains useful prior context
- ✅ **Improved Responses**: Fetches relevant older messages
- ✅ **Natural Experience**: Feels more human-like

---

## 🛠️ Core Implementation Overview

We build four key functions:
1. `load_history()` - Load all past messages into memory (conversation_history[] list is the memory here )
2. `add_to_history(role, content)` - Add new messages
3. `save_history()` - Persist current memory to file
4. `get_relevant_history(query)` - Retrieve relevant past messages

---

## 📦 Setup

```python
import json
import os
from datetime import datetime

HISTORY_FILE = "conversation_history.json"
conversation_history = []

```

##  load_history()

* Checks if the history file exists

* If yes, opens it and loads the JSON content into the global conversation_history list

##  add_to_history(role, content)

* Appends a new dictionary (message) to the conversation_history list

    * The message includes:

    * role: "user" or "assistant"

    * content: The actual text

    * timestamp: The current date and time

* Calls save_history() to persist changes


## save_history()

* Opens the file in write mode ('w')

* Dumps the entire conversation_history list as formatted JSON

* Uses indent=2 for readability

##  get_relevant_history(query, max_messages=5)

Converts the query into a lowercase set of words (query_words)

* Iterates over all past messages

* For each message:

    * Tokenizes its content into words

    * Counts how many words overlap with the query

    * Adds (overlap_score, message) to the list if overlap > 0

* Sorts the messages by overlap score, descending

* Returns the top max_messages messages (default = 5), stripping the score