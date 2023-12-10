# Have two models converse with each other
# Specialized to Mistral (prompts are different)

# Without a grammar, controlling the output is not possible.
# Each large language model (LLM) generates a complete conversation.
# However, by specifying that each LLM represents a single author responding to another, the output appears to be more controlled.

import os
import re
import generic_utils as u
from langchain.callbacks.manager import CallbackManager

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from langchain.chains import LLMChain
from langchain.llms import LlamaCpp

# from langchain.prompts import PromptTemplate
# from llama_cpp.llama import Llama
from llama_cpp.llama_grammar import LlamaGrammar

# not used on mac
os.environ["CUBLAS_LOGINFO_DBG"] = "1"

# Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
callback_manager = CallbackManager([])
# ----------------------------------------------------------------------

n_gpu_layers = 20  # Change this value based on your model and your GPU VRAM pool.
n_batch = 128  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

LLM_MODELS = os.environ["LLM_MODELS"]

# with open("json.gbnf", "r") as file:
# with open("json_arr.gbnf", "r") as file:
with open("json_converse.gbnf", "r") as file:
    grammar_text = file.read()
grammar = LlamaGrammar.from_string(grammar_text)


# Make sure the model path is correct for your system!
def myLlamaCpp(model: str):
    """
    Create an instance of LlamaCpp with the given model path.

    Args:
        model (str): The path to the LlamaCpp model.

    Returns:
        LlamaCpp: An instance of the LlamaCpp class.
    """
    llm = LlamaCpp(
        model_path=model,
        n_gpu_layers=n_gpu_layers,
        n_ctx=4096,
        stop=[],
        # stop=["</s>"],  # If used, the message will stop early
        max_tokens=1000,
        n_threads=8,
        temperature=0.8,  # also works with 0.0 (0.01 is safer)
        f16_kv=True,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=False,
    )
    return llm


modelA = LLM_MODELS + "mistral-7b-instruct-v0.1.Q3_K_M.gguf"
modelB = LLM_MODELS + "mistral-7b-instruct-v0.1.Q3_K_M.gguf"

llmA = myLlamaCpp(modelA)
llmB = myLlamaCpp(modelB)

authorA = "Stephen Hawking"
authorB = "Lee Smolin"

subject = """The topic of discussion is quantum loop gravity and string theory as approaches to 
          merge quantum mechanics with the theory of relativity"""

additional_context_authorA = f"""The answers should be diversified, the conversation engaging, and repetitions should be minimized. Each answer should take into account the conversation up to this point. Your reply to {authorB} is in JSON format: {{"Interlocutor": {authorA}, "Reply": xxx}}, where `xxx` if the reply by {authorB}"""

additional_context_authorB = f"""The answers should be diversified, the conversation engaging, and repetitions should be minimized. Each answer should take into account the conversation up to this point. Your reply to {authorA} is in JSON format: {{"Interlocutor": {authorB}, "Reply": xxx}}, where `xxx` is the reply by {authorB}"""

promptA = f"""You are {authorA}. {subject}. {additional_context_authorA}. You'll be answering comments by {authorB}.  Initiate the conversation with a question to {authorB}."""

promptB = f"""You are {authorB}. {subject}. {additional_context_authorB}. You'll be answering comments by {authorA}. """

# Remove 2 or more consecutive spaces.
promptA = re.sub(" {2,}", "", promptA)
promptB = re.sub(" {2,}", "", promptB)

msgsA = u.MistralMessages()
msgsB = u.MistralMessages()

msgsA.add_instruction("system", promptA)
msgsB.add_instruction("system", promptB)
msgsA.add(authorA, "")


conversation = u.Conversation(authorA, authorB, llmA, llmB, msgsA, msgsB, grammar)
# conversation.print_both_contexts()
conversation.multi_turn(3)
quit()
