from gpt4all import GPT4All
from pathlib import Path

model_folder = Path.home() / "data/llm_models"

models = [
    'mistral-7b-instruct-v0.1.Q4_0.gguf',  # 0
    'gpt4all-falcon-newbpe-q4_0.gguf',     # 1
    'orca-2-7b.Q4_0.gguf',                 # 2
    'mpt-7b-chat-newbpe-q4_0.gguf',        # 3
    'orca-mini-3b-gguf2-q4_0.gguf'         # 4
]

def run_models(prompts):
    for model in models:
        llm = GPT4All(model, model_path=model_folder, allow_download=True)
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

#----------------------------------------------------------------------
def get_llm(model_id):
    return GPT4All(models[model_id], model_path=model_folder, allow_download=False)
#----------------------------------------------------------------------

