# 🧠 Context Window Optimization (with Python Example)

## ✅ What is Context Window Optimization?

Selecting the most relevant subset of conversation history or documents to stay within the model’s token limit, typically:
- 4096 tokens (GPT-3.5)
- 8192 / 128000 tokens (GPT-4)

## 🔥 Why Optimize?

| Benefit       | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| Performance   | Lower latency, cost                                                         |
| Relevance     | More accurate responses from focused context                                |
| Scalability   | Enables long interactions without truncation                                |

---

## 🧪 Code Example: Merging and Deduplicating Messages

```python

# Final context window (remove duplicate contents, keep last one)
context_msgs = list({msg["content"]: msg for msg in recent_msgs + relevant_msgs}.values())

# Example data:
recent_msgs = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello"},
    {"role": "user", "content": "How are you?"}
]

relevant_msgs = [
    {"role": "assistant", "content": "Hello"},
    {"role": "assistant", "content": "I can help you"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "Nice to meet you"}
]

### STEP 1: Combine both lists:
#  combined = recent_msgs + relevant_msgs

combined = [
    {"role": "user", "content": "Hi"},                    # 0
    {"role": "assistant", "content": "Hello"},            # 1
    {"role": "user", "content": "How are you?"},          # 2
    {"role": "assistant", "content": "Hello"},            # 3 (duplicate content!)
    {"role": "assistant", "content": "I can help you"},   # 4
    {"role": "user", "content": "How are you?"},          # 5 (duplicate content!)
    {"role": "assistant", "content": "Nice to meet you"}  # 6
]

### STEP 2: Dictionary comprehension

{msg["content"]: msg for msg in combined}

""" This creates a dictionary where:

key = the message content (string)

value = the full message object (the dict itself)

Now, let’s simulate the dictionary being built step-by-step:

Index	content (key)	value (whole message)	Replaces?	Dictionary So Far
0	"Hi"	{"role": "user", "content": "Hi"}	No	{ "Hi": {...} }
1	"Hello"	{"role": "assistant", "content": "Hello"}	No	{ "Hi": {...}, "Hello": {...} } """

# Output:
{
  "Hi":               {"role": "user", "content": "Hi"},
  "Hello":            {"role": "assistant", "content": "Hello"},        # latest version
  "How are you?":     {"role": "user", "content": "How are you?"},     # latest version
  "I can help you":   {"role": "assistant", "content": "I can help you"},
  "Nice to meet you": {"role": "assistant", "content": "Nice to meet you"}
}


### STEP 3: .values() gives all values of the dictionary
dict_obj.values()

# Output:

dict_values([
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I can help you"},
    {"role": "assistant", "content": "Nice to meet you"}
])


### STEP 4: Convert to a list

list(...)
Now you get a final list:

context_msgs = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I can help you"},
    {"role": "assistant", "content": "Nice to meet you"}
]

```