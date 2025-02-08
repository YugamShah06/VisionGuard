from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Meditron-7B or Meditron-70B model
model_name = "epfl-llm/meditron-7b"  # For Meditron-7B
# model_name = "epfl-llm/meditron-70b"  # Uncomment this line for Meditron-70B
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Function to generate a prescription
def generate_prescription(condition, age, allergies):
    sys_message = ''' 
    You are an AI Medical Assistant trained on a vast dataset of health information. Please provide a doctor's note or prescription
    based on the patient's condition, age, and any known allergies. If necessary, include a disclaimer that the user should consult a healthcare provider.
    '''   
    
    # Constructing the question based on inputs
    question = f'''The patient has been diagnosed with {condition}. 
    The patient's age is {age}, and their known allergies are {allergies}. 
    Please generate a prescription or doctor's note for this condition.'''
    
    # Prepare the full prompt
    prompt = f"{sys_message}\n\n{question}"

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    # Generating the response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)

    # Extracting the generated response
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    return response_text

# Example usage
detected_condition = "conjunctivitis"
age = 25
allergies = "None"

# Generate prescription
prescription = generate_prescription(detected_condition, age, allergies)
print(prescription)
