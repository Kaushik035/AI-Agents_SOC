```python
entities = {}

```


- This is your memory storage.
- It's a Python dictionary that will map each entity (like `"Python"`) to the message where it appeared and a timestamp.

Example after storing something:
```python
entities = {
    "Python": {
        "context": "Tell me about Python.",
        "timestamp": "2025-06-29T09:30:22"
    }
}
```

```python
def extract_entities(text):
"""Store capitalized words as entities."""
words = text.split()
for word in words:
if word.istitle() and len(word) > 2:
entities[word] = {
"context": text,
"timestamp": datetime.now().isoformat()
}
```


#### 🔍 Explanation:

1. **text.split()** breaks input into individual words.
2. **word.istitle()** checks if a word starts with a capital letter and rest are lowercase (e.g., "Python", "India").
3. **len(word) > 2** filters out small words like “It”, “Is”, etc.

Each valid word is added to `entities` with:
- `"context"`: the sentence it came from
- `"timestamp"`: when it was mentioned

#### 🧪 Example:

Call this:
```python
extract_entities("Tell me about Python and Java.")

# Now entities will hold:

{
  "Tell": {"context": "Tell me about Python and Java.", ...},
  "Python": {"context": "Tell me about Python and Java.", ...},
  "Java": {"context": "Tell me about Python and Java.", ...}
}

```

```python
def get_entity_context(entity):
"""Retrieve entity context if mentioned."""
return entities.get(entity, {}).get("context", "")
```


- Looks up the entity in `entities` dictionary.
- If the entity exists, returns the stored `"context"` value.
- If not found, returns an empty string `""`.

#### 🧪 Example:
```python
get_entity_context("Python")
```

Returns:
```python
"Tell me about Python and Java."
```

---

## 🔄 Summary Table

| Component        | Purpose                                       |
|------------------|-----------------------------------------------|
| `extract_entities(text)` | Adds capitalized words to memory |
| `get_entity_context(entity)` | Retrieves memory context |
| `context.append(...)` | Adds memory to model prompt |
| `call_openai(...)` | Sends full prompt to model |

---