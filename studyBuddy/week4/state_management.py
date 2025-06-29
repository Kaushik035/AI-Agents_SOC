import os
import json
from datetime import datetime

import spacy
import tiktoken
from shared.newOpenAI import openai

#   Global config & helpers

nlp          = spacy.load("en_core_web_sm")
encoding     = tiktoken.encoding_for_model("gpt-3.5-turbo")

HISTORY_FILE        = "SBconversation_history.json"
MAX_HISTORY_TOKENS  = 3000        # hard ceiling for prompt context
MAX_RECENT_MESSAGES = 3           # how many latest turns to always try to keep
MAX_RELEVANT_MSGS   = 3          # max candidates picked via relevance

conversation_history: list[dict] = []
entities: dict[str, dict]        = {}



#  Conversation-history persistence
def load_history() -> None:
    """Load history from disk (called once at startup)."""
    global conversation_history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            conversation_history = json.load(f)

def save_history() -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, indent=2)

def add_to_history(role: str, content: str) -> None:
    conversation_history.append(
        {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
    )
    save_history()

#   Token helpers
def get_token_count(text: str) -> int:
    """Accurate token count via tiktoken."""
    return len(encoding.encode(text))

#  Relevance search (keyword overlap)
def get_relevant_history(query: str, max_messages: int = MAX_RELEVANT_MSGS) -> list[dict]:
    """Return up to *max_messages* past messages scored by word overlap."""
    query_words   = set(query.lower().split())
    scored: list[tuple[int, dict]] = []

    for msg in conversation_history:
        msg_words = set(msg["content"].lower().split())
        overlap   = len(query_words & msg_words)
        if overlap > 0:
            scored.append((overlap, msg))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [msg for _, msg in scored[:max_messages]]




#   Context-window optimisation (recent + relevant)
def get_optimized_context(query: str,
                          max_total_tokens: int = MAX_HISTORY_TOKENS
                          ) -> list[dict]:
    """
    Build a prompt context that includes:
      • last N recent turns        (MAX_RECENT_MESSAGES)
      • top-scoring relevant turns (keyword overlap)
    De-duplicates by content and trims to *max_total_tokens*.
    """
    recent_msgs   = conversation_history[-MAX_RECENT_MESSAGES:]
    relevant_msgs = get_relevant_history(query)

    seen: set[str] = set() # Track unique message content
    combined: list[dict] = []

    # Preserve order: recent first, then relevant
    for msg in recent_msgs + relevant_msgs:
        if msg["content"] not in seen:
            combined.append(msg)
            seen.add(msg["content"])

    # Token-budget pruning
    total = 0
    final_context: list[dict] = []
    for msg in combined:
        tok = get_token_count(msg["content"])
        if total + tok <= max_total_tokens:
            final_context.append(msg)
            total += tok
        else:
            break
    return final_context

# Below function is now not used in the code, but kept for reference, if we want the prompt template to include the full history
def get_history_string() -> str:
    """
    Human-readable string of the *token-bounded* history
    (used only for display in your prompt template).
    """
    return "\n".join(f"{m['role']}: {m['content']}" for m in get_optimized_context("", MAX_HISTORY_TOKENS))


#   Entity tracking via spaCy
def extract_entities(text: str) -> None:
    """Store unseen named entities with minimal context."""
    doc = nlp(text)
    for ent in doc.ents:
        entities.setdefault(ent.text, {
            "type": ent.label_,
            "context": text,
            "timestamp": datetime.now().isoformat()
        })

def get_entity_context(entity: str) -> str:
    return entities.get(entity, {}).get("context", "")



#   Summarisation for very long histories

def summarize_history() -> str:
    """
    Abstractive summary if full transcript exceeds ~200 tokens.
    Returns the raw history if already short.
    """
    full_text = " ".join(m["content"] for m in conversation_history)
    if get_token_count(full_text) < 200:
        return full_text

    prompt = (
        "Summarize the following conversation in under 100 words:\n"
        + full_text
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp["choices"][0]["message"]["content"]
