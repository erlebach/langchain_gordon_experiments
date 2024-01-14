"""
# Question for the capital of France in two ways: 
# Best for instruction models. 
- What is the capital of France?
# Best for word completion model
- The capital of France is

Next, a more complex case. 
"What is the capital of France? Keep your answer"

"""
from gpt4all import GPT4All
import utils as u


# Even with `allow_download=True`, the model model is not downloaded if in the stated location.

prompt =  "GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:"
prompt =  "User: Hello<|end_of_turn|>Assistant: Hi<|end_of_turn|>User: How are you today?<|end_of_turn|>Assistant:"

prompts = []
prompts.append( "What is the capital of France?" )
prompts.append( "The capital of France is" )
prompts.append( "What is the capital of France? Keep your answer" )

# None of the models generated a continuation response. They all acted as instruct models
prompts = ["I completed all my tasks. I am now available to "]
prompts = ["I completed all my tasks. What is next?"]
# Best model will answer to the point. 
prompts = ["I completed all my tasks. Do you have another task for me?"]
prompts= ["Tell me the basic axioms of quantum mechanics"]

for i, prompt in enumerate(prompts): 
    print(f"Prompt [{i}]: {prompt}")
print()

#u.run_models(prompts)
llm = u.get_llm(model_id=0)
reply = llm.generate(prompts[0], streaming=False, max_tokens=50, temp=0.01)
print("reply: ", reply)
reply = llm.generate(prompts[0], streaming=True, max_tokens=50, temp=0.01)
print(f"reply: {reply}")

# Notice that the tokens are not words in general. 
for token in reply:
    print(f"Token: '{token}'")

"""
For example
Tokens: 
Token: ' or'
Token: ' measured'
Token: ','
Token: ' at'
Token: ' which'
Token: ' point'
Token: ' it'
Token: ' coll'
Token: 'aps'
Token: 'es'
Token: ' into'

A word list would be: 
Word: 'or'
Word: 'measured'
Word: 'measured'
Word: ','
Word: 'at'
Word: 'which'
Word: 'point'
Word: 'it'
Word: 'collapses'  # Was 3 tokens
Word: 'into'
"""
