import os
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 20  # Change this value based on your model and your GPU VRAM pool.
n_batch = 128  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

LLM_MODELS = os.environ['LLM_MODELS']


# Make sure the model path is correct for your system!
def myLlamaCpp(model):
    llm = LlamaCpp(
        model_path=model,
        n_gpu_layers=n_gpu_layers,
        n_ctx=2048,
        stop=[],
        max_tokens=1000,
        n_threads=8,
        temperature=0.4,  # also works with 0.0 (0.01 is safer)
        f16_kv=True,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=False,  
    )
    return llm

modelA = LLM_MODELS + "mistral-7b-instruct-v0.1.Q3_K_M.gguf"
llmA = myLlamaCpp(modelA)

contextA = "You are Sir Isaac Newton, believing the idea of an absolute space background and the concept of absolute time. Please give your thoughts on a quantum  theory of gravity."

llmA(contextA)
