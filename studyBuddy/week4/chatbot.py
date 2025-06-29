from shared.newOpenAI import openai
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tools import search_tavily, search_wikipedia, calculate

# ░░  State-management helpers

from state_management import (
    load_history, add_to_history, extract_entities,
    get_entity_context, get_optimized_context,
    summarize_history, conversation_history, entities
)

# ░░  RAG core
def load_and_chunk_document(file_path="studyBuddy/week4/notes/my_note.md"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise ValueError("Document is empty.")
    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 10]
    return chunks

def generate_embeddings(chunks):
    embedder  = SentenceTransformer("all-MiniLM-L6-v2")
    embeds    = embedder.encode(chunks, convert_to_numpy=True)
    return embeds, embedder

def create_faiss_index(embeddings):
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_chunks(question, embedder, index, chunks, k=2):
    q_emb    = embedder.encode([question], convert_to_numpy=True)
    _, idx   = index.search(q_emb, k)
    return [chunks[i] for i in idx[0]]

# ░░  Tool selection & execution
def select_tool(query: str):
    q = query.lower()
    if "search" in q or "latest" in q:
        return "tavily"
    if "wiki" in q or "who is" in q or "who was" in q:
        return "wikipedia"
    if any(op in q for op in ["+", "-", "*", "/", "calculate"]):
        return "calculator"
    return "none"

def get_tool_output(tool: str, query: str):
    if tool == "tavily":
        return search_tavily(query)
    if tool == "wikipedia":
        return search_wikipedia(query)
    if tool == "calculator":
        expr = query.lower().split("calculate", 1)[-1].strip() if "calculate" in query.lower() else query
        return calculate(expr)
    return "No external tool used."

# ░░  OpenAI call (context-aware)
def call_openai(query: str,
                context_msgs: list[dict],
                notes: str,
                tool_output: str) -> str:
    """
    • context_msgs: optimized recent+relevant history + any system inserts
    • notes/tool_output: RAG evidence
    """
    messages = [ {"role": m["role"], "content": m["content"]} for m in context_msgs ]

    # Inject RAG info as a single system message
    system_content = (
        f"Relevant notes:\n{notes}\n\n"
        f"External tool output:\n{tool_output}\n\n"
        "Now answer the user's question clearly. "
        'If nothing in notes or tool output helps, reply with '
        '"I cannot find the answer in the provided notes or tools."'
    )
    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user",  "content": query})

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return resp["choices"][0]["message"]["content"]


def run_chatbot():
    # 1️⃣  Load persistent history
    load_history()

    # 2️⃣  Prep RAG index
    chunks, embedder, index = None, None, None
    try:
        chunks = load_and_chunk_document()
        embeddings, embedder = generate_embeddings(chunks)
        index = create_faiss_index(embeddings)
    except Exception as e:
        print("Note loading RAG document failed:", e)

    print("Welcome to the Study Buddy! Type 'quit' to exit.")

    while True:
        query = input("\nYou: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break

        # 3️⃣  Update state trackers
        add_to_history("user", query)
        extract_entities(query)

        # 4️⃣  Build optimized context
        context = get_optimized_context(query)
        print("Context size:", len(context))
        print("Context messages before entity notes:", context)

        # 5️⃣ Entity note (optimized)
        mentioned_entities = [
            ent for ent in entities.keys() if ent.lower() in query.lower()
        ]

        for ent in mentioned_entities:
            entity_context = get_entity_context(ent)
            if entity_context:
                context.append({
                    "role": "system",
                    "content": f"Note: User mentioned '{ent}': {entity_context}"
                })

        if mentioned_entities.__len__() > 0:
            print("Context messages after entity notes:", context)


        # 6️⃣  Summarize if history long
        if len(conversation_history) > 10:
            print("History is long, summarizing...")
            summary = summarize_history()
            context.append({"role": "system", "content": f"Summary: {summary}"})

        # 7️⃣  RAG + Tool pipeline
        tool       = select_tool(query)
        tool_out   = get_tool_output(tool, query)

        rag_notes  = ""
        if chunks and embedder and index:
            rag_notes = "\n\n".join(retrieve_chunks(query, embedder, index, chunks))

        # 8️⃣  Call LLM
        answer = call_openai(query, context, rag_notes, tool_out)
        print("\nAssistant:", answer)

        # 9️⃣  Persist assistant reply
        add_to_history("assistant", answer)

# ──────────────────────────────────────────────
if __name__ == "__main__":
    run_chatbot()
