from shared.newOpenAI import openai

# Load the Markdown notes
with open("studyBuddy/week1/notes/my_notes.md", "r", encoding="utf-8") as f:
    notes_content = f.read()

# Prompt template
prompt_template = """
You are an AI Study Assistant. Answer the following question based ONLY on the provided notes.
If the answer is not in the notes, state 'I cannot find the answer in the provided notes.'

Notes:
{notes_content}

Question: {question}
Answer:
"""

while True:
    question = input("\nEnter your question (or type 'quit' to exit): ")
    if question.lower() == "quit":
        print("Goodbye!")
        break

    prompt = prompt_template.format(notes_content=notes_content, question=question)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["choices"][0]["message"]["content"]
    print("\nAnswer:", answer)
