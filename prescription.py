import requests
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
import json

# Replace these with actual API keys and URLs
API_KEY = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjVkZTdiZDhkOTg0Njg2NDEyZmI4OGVkYWY0Mzk4N2NkIiwiY3JlYXRlZF9hdCI6IjIwMjQtMTAtMjRUMDk6NDE6MDYuNzAyNTI2In0.5tnLCfJ7ZvtTx5zdeG-6I-cmZuTTY6flE0e59mRxwJU'
API_URL = 'https://llm.monsterapi.ai/v1/generate'

# Load a pre-trained model for feature extraction (you can replace with a custom model)
def load_model():
    model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    return model

# Preprocess the eye image (resize and normalize)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to 224x224 for EfficientNet
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Preprocess for EfficientNet
    return image

# Extract features from the image using the pre-trained model
def extract_features(model, image):
    features = model.predict(image)
    return features

# Function to generate a prescription using the Monster API
def generate_prescription(features, patient_info):
    # Convert the features to a list or serializable format
    features = features.tolist()

    data = {
        "image_features": features,
        "patient_info": patient_info,
        "model": "llama-medical",  # Specify the LLaMA model name trained on medical data
        "condition": "conjunctivitis"  # Set the condition
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        # Make the request to the Monster API
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response to extract the prescription
        result = response.json()
        prescription = result.get('prescription', 'No prescription generated')

        return prescription

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Error occurred: {err}")

# Example usage
if __name__ == "__main__":
    # Load the image and model
    model = load_model()
    image_path = 'eye-condition-detection-deep-learning-main\scraping\conjunctivitis bing\C0269241-Viral_conjunctivitis-SPL.width-1534.jpg'  # Replace with your image path
    image = preprocess_image(image_path)
    
    # Extract features from the image
    features = extract_features(model, image)

    # Define patient information
    patient_info = {
        "age": 25,
        "medical_history": ["no_known_allergies"],
        "duration_of_symptoms": "3 days"
    }

    # Generate the prescription
    prescription = generate_prescription(features, patient_info)
    print("Generated Prescription:", prescription)
