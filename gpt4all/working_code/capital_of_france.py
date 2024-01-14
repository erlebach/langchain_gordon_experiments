"""
# Question for the capital of France in two ways: 
# Best for instruction models. 
- What is the capital of France?
# Best for word completion model
- The capital of France is

Next, a more complex case. 
"What is the capital of France? Keep your answer"

"""
from pathlib import Path
from gpt4all import GPT4All

models = [
    'mistral-7b-instruct-v0.1.Q4_0.gguf',  # 0
    'gpt4all-falcon-newbpe-q4_0.gguf',     # 1
    'orca-2-7b.Q4_0.gguf',                 # 2
    'mpt-7b-chat-newbpe-q4_0.gguf',        # 3
    'orca-mini-3b-gguf2-q4_0.gguf'         # 4
]

model_folder = Path.home() / "data/llm_models"

# Even with `allow_download=True`, the model model is not downloaded if in the stated location.

prompt =  "GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:"
prompt =  "User: Hello<|end_of_turn|>Assistant: Hi<|end_of_turn|>User: How are you today?<|end_of_turn|>Assistant:"

def run_models(prompts):
    for model in models:
        # Only device='cpu' works. 
        llm = GPT4All(model, model_path=model_folder, allow_download=True, verbose=True)  
        print(f"==========> {model=}") 
        print(f"{llm.config['systemPrompt']=}")
        print(f"{llm.config['promptTemplate']=}")
        print()

        for i, prompt in enumerate(prompts):
            the_prompt = "User: {my_prompt}<|end_of_turn|>Assistant:"
            the_prompt = the_prompt.format(my_prompt=prompt)
            print("the_prompt: ", the_prompt)
            output = llm.generate(the_prompt, max_tokens=20, temp=0.01)
            print(f"[{i}]: {output}")
            print("--------------------------------------")
        print("=============================================================================")

prompts = []
prompts.append( "What is the capital of France?" )
prompts.append( "The capital of France is" )
prompts.append( "What is the capital of France? Keep your answer" )

# None of the models generated a continuation response. They all acted as instruct models
prompts = ["I completed all my tasks. I am now available to "]
prompts = ["I completed all my tasks. What is next?"]
prompts = ["I completed all my tasks. Do you have another task for me??"]

for i, prompt in enumerate(prompts): 
    print(f"Prompt [{i}]: {prompt}")
print()

run_models(prompts)

