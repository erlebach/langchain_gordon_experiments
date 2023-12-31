# Zephyr
I am running the pretrained version of Zephyr on my CPU on the Mac (not the GPU). Here are first results.
What is amazing is how close the structure of these sentences are to one another.
The code took about 102 seconds on my mac!!! Took 928s on class7 (ran it twice with time cmd). 
It did not actually take 928 sec so I don't understand.  With flash_attn= True, took 945 sec according to bash cmd "time". 
Each inference cost 34sec on the CPU on class7. Measured with  two calls to time(). 
The cost to load the 7B model is 45sec. The model is not quantized. 

On my mac: 49 sec to load the model (CPU) and 33 sec for each inference.  (see below)
So my mac is approx same speed as linux machine? Strange. 

On an empty classroom machine: class18, 160 sec to load shards, 19 sec for inference (makes more sense) on the CPU. 
2nd run, same results. torch confirms there is a GPU. HOwever, the model was run on the CPU since the model was
not transferred to the GPU using to.device('cuda'). nvidia-smi does not work. 
Error: "(langchain-monorepo-py3.11) ➜  zephyr_experiments git:(main) ✗ nvidia-smi
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 545.23"

I cannot run Mistral on my mac gpu using llama.cpp (I could not get it to compile). No luck with llama-cpp-python either. Mistral has another issue: it cannot be run without flash attention, which requires CUDA. Zephyr can be run with and without flash attention (there is an argument in from_pretrained(....) that controls that.

I will try to run this same setup on the classroom machines this weekend, hopefully.
Create a GPU branch. In that way, I can maintain the same repo on different systems. 

-----------------------------------------
>>>> From Zephyr:

Are you happy?

I’m not talking about the kind of happiness that comes from a good day, a great meal, or a fun night out with friends. I’m talking about the kind of happiness that comes from within,
Are you happy?

I’m not talking about the fleeting happiness that comes from a good meal, a funny movie, or a great workout. I’m talking about the deep, abiding happiness that comes from knowing who you
Are you happy?

I’m not talking about the kind of happiness that comes from a good day, a promotion, or a new relationship. I’m talking about the kind of happiness that comes from within, the kind that doesn
(langchain-monorepo-py3.10) (base) ➜  phi_experiments git:(main) ✗

# Mistral

# Phi1.5
----------------------------------------------------------------------
2023-12-10
Some ideas from Perplexity: 
- Use only a single context, called conversation. 
- Inject instructions at the end of the the conversations to increase the likelihood
  of the LLM replying correctly. 
``` python
  def converse(model1, model2, start_prompt):
    conversation = [start_prompt]
    while True:
        # Model 1 generates a response based on the conversation so far
        prompt1 = f"AuthorA: {conversation[-1]}\nAuthorB: [INST]What would AuthorA say in response to AuthorB?[/INST]"
        response1 = model1.generate_response(prompt1)
        conversation.append(response1)
        
        # Model 2 generates a response based on the conversation so far
        prompt2 = f"AuthorA: {response1}\nAuthorB: [INST]What would AuthorB say in response to AuthorA?[/INST]"
        response2 = model2.generate_response(prompt2)
        conversation.append(response2)
        
        # You can add a condition to break the loop if the conversation reaches a certain length
        if len(conversation) > 10:
            break
    return conversation
```
----------------------------------------------------------------------
2023-12-11
More details on the parameters: 
https://forum.cloudron.io/topic/8872/serge-llama-made-easy-self-hosted-ai-chat/2

Additional parameters to llama.cpp
 repeat_penalty=1.1, frequency_penalty=0.0, presence_penalty=0.0,
----------------------------------------------------------------------
