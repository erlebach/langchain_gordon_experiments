from flask import Flask, request, jsonify
import redis  # database that serves as cache
from gpt4all import GPT4All
import os

app = Flask(__name__)

# redis runs on the same machine as the server
cache = redis.Redis(host='redis', port=6379)

#LLM_MODELS=os.environ['LLM_MODELS']
LLM_MODELS="/app/llm_models"

# Specify the path to the model folder
#model_path = '/Users/erlebach/data/llm_models/'
model_path = LLM_MODELS
model = 'mistral-7b-instruct-v0.1.Q4_0.gguf'

# Initialize the GPT4All instance with the specified model path
gpt_instance = GPT4All(model, model_path=model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    prompt = data.get('prompt', '')
    
    # Perform prediction using GPT4All
    response = gpt_instance.generate(prompt=prompt)
    
    # Return the prediction as a JSON response
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4891)
    #app.run(debug=True, host='127.0.0.1', port=4892)
