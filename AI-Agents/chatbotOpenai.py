from newOpenAI import openai


def create_prompt(user_input):
    prompt = f"User: {user_input}\nAssistant:"
    return prompt


def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def process_response(generated_text):

    # Optional: Cut off incomplete sentences (simple approach)
    if not generated_text.endswith("."):
        last_period = generated_text.rfind(".")
        if last_period != -1:
            generated_text = generated_text[:last_period + 1]
    return generated_text

# Main chatbot loop
def run_chatbot():
    print("Welcome to the AI Chatbot! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        prompt = create_prompt(user_input)
        raw_response = generate_response(prompt)
        final_response = process_response(raw_response)
        print(f"Assistant: {final_response}")

# Start the chatbot
run_chatbot()