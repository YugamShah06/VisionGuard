# Install required libraries
# !pip install transformers accelerate torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the Llama 2 7B Chat model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Set up the pipeline
chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define the conversation message (user input)
messages = [
    {"role": "user", "content": "Who are you?"},
]

# Create the prompt for the conversation
prompt = f"User: {messages[0]['content']}\nAssistant:"

# Generate the response from the model
response = chat_pipeline(prompt, max_length=100, num_return_sequences=1, truncation=True)

# Print the response
print(response[0]["generated_text"])