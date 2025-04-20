# llama_model.py
from llama_cpp import Llama

def load_model(model_path):
    """Load the Llama model from the given path"""
    llama_model = Llama(model_path=model_path)
    return llama_model

def generate_response(model, prompt):
    """Generate a response from the Llama model"""
    response = model(prompt)
    return response["text"]
