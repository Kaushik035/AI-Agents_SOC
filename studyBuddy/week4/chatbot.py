import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from shared.newOpenAI import openai

# Tool-chaining API (handles validation + dependency management)
from tools_chain import run_tool_chain, fallback_openai

from state_management import (
    load_history,
    add_to_history,
    extract_entities,
    get_entity_context,
    get_optimized_context,
    summarize_history,
    save_history,
    conversation_history,
    entities,
)

USER_LEVEL = "high_school"

from persona import (
    build_persona_system_prompt,
    check_ethical_compliance,      
)

#  ░░  RAG core – helper functions
def load_and_chunk_document(file_path="studyBuddy/week4/notes/my_note.md"):
    """Load notes and split into non-empty paragraphs."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError("Document is empty.")

    chunks = [para.strip() for para in text.split("\n\n") if len(para.strip()) > 10]
    return chunks


def generate_embeddings(chunks):
    """Create sentence-transformer embeddings for note chunks."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embedder.encode(chunks, convert_to_numpy=True)
    return vectors, embedder


def create_faiss_index(embeddings):
    """Put embeddings into a simple FAISS L2 index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def retrieve_chunks(question, embedder, index, chunks, k=2):
    """Return top-k relevant note chunks for the question."""
    question_vec, _ = embedder.encode([question], convert_to_numpy=True), None
    _, ids = index.search(question_vec, k)
    return [chunks[i] for i in ids[0]]


def call_openai(query, context_msgs, rag_notes, tool_output):
    """
    Build messages list and call OpenAI, injecting
      • persona-aware system prompt
      • RAG notes
      • tool output
    """
    #1 build persona system prompt (domain + style)
    persona_prompt = build_persona_system_prompt(query, USER_LEVEL)

    #2 assemble message list
    messages = [{"role": "system", "content": persona_prompt}]
    messages += [{"role": m["role"], "content": m["content"]} for m in context_msgs]

    # evidence block
    evidence_block = (
        f"Relevant notes:\n{rag_notes}\n\n"
        f"External tool output:\n{tool_output}\n\n"
        "Answer the user's question clearly. "
        'If the notes and tool output do not provide enough information, say '
        '"I cannot find the answer in the provided notes or tools."'
    )
    messages.append({"role": "system", "content": evidence_block})
    messages.append({"role": "user", "content": query})

    #3 call OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response["choices"][0]["message"]["content"]



def run_chatbot():
    """Run the interactive Study Buddy."""
    # 1. Load past conversation history
    load_history()

    # 2. Build RAG index (if notes file exists)
    chunks = embedder = index = None
    try:
        chunks = load_and_chunk_document()
        embeddings, embedder = generate_embeddings(chunks)
        index = create_faiss_index(embeddings)
    except Exception as exc:  # noqa: BLE001
        print("RAG disabled:", exc)

    print("Welcome to the Study Buddy!  (type 'quit' to exit)\n")

    while True:
        query = input("You: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break

        # 3. Update memory and entity store
        add_to_history("user", query)
        extract_entities(query)

        # 4. Build optimized context (recent + relevant)
        context = get_optimized_context(query)

        # 4-a: Add entity memories if re-mentioned
        mentioned_entities = [
            ent for ent in entities.keys()           # every stored entity
            if ent.lower() in query.lower()          # that appears in the user query   
            ]

        for ent in mentioned_entities:
            entity_context = get_entity_context(ent)
            if entity_context:
                context.append(
                    {
                        "role": "system",
                        "content": f"Note: User mentioned '{ent}': {entity_context}",
                    }
                )

        # 4-b: Add summary if conversation is long
        if len(conversation_history) > 10:
            print("History is long, summarizing…")
            summary = summarize_history()
            context.append(
                {
                    "role": "system",
                    "content": f"Summary: {summary}",
                }
            )

        # 5. Run the tool chain
        tool_output = run_tool_chain(query) # we have added fallback_openai in tools_chain.py so there will always some output

        # 6. Retrieve RAG notes
        rag_notes = ""
        if chunks and embedder and index:
            rag_notes = "\n\n".join(retrieve_chunks(query, embedder, index, chunks))

        # 7. Call OpenAI with full context
        answer = call_openai(query, context, rag_notes, tool_output)

        # 7-a. Ethical compliance check
        ok, msg = check_ethical_compliance(answer)
        if not ok:
            answer = (
                "⚠️ Sorry, I can’t provide that response due to ethical concerns "
                f"({msg}). Please rephrase your question."
            )
        print("\nAssistant:", answer, "\n")

        # 8. Conditionally store assistant reply
        if answer.strip() == "I cannot find the answer in the provided notes or tools.":
            # Remove last user message too
            if conversation_history and conversation_history[-1]["role"] == "user":
                conversation_history.pop()
                save_history()
        else:
            add_to_history("assistant", answer)



if __name__ == "__main__":
    run_chatbot()
