from transformers import pipeline

# Load GPT-2 for text generation
generator = pipeline("text-generation", model="gpt2")

def create_prompt(user_input):
    prompt = f"User: {user_input}\nAssistant:"
    return prompt

def generate_response(prompt):
    # Generate text with GPT-2
    response = generator(prompt, max_length=50, num_return_sequences=1)
    # Extract the generated text
    generated_text = response[0]["generated_text"]
    return generated_text

def process_response(generated_text):
    # Split at "Assistant:" and take the response part
    assistant_response = generated_text.split("Assistant:")[1].strip()
    # Optional: Cut off incomplete sentences (simple approach)
    if not assistant_response.endswith("."):
        last_period = assistant_response.rfind(".")
        if last_period != -1:
            assistant_response = assistant_response[:last_period + 1]
    return assistant_response

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