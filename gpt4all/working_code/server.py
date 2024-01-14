import openai
from gpt4all import GPT4All
from pathlib import Path

# Set the API base to the local server address
openai.api_base = "http://localhost:4891/v1"
openai.api_key = "not needed for a local LLM"

# Specify the path to the folder containing the model
model_folder = Path.home() / "data/llm_models"
model_file = "mistral-7b-instruct-v0.1.Q4_0.gguf"

# Initialize the GPT4All model with the specified folder
model = GPT4All(model_file, model_folder)

# Start the server with the loaded model
model.serve()
