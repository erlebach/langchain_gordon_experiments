import numpy as np
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import gpt4all

from langchain_community.embeddings import GPT4AllEmbeddings

import langchain_community.embeddings as embed

gpt4all_embd = GPT4AllEmbeddings()
# model_path="/Users/erlebach/data/llm_models/codeninja-1.0-openchat-7b.Q4_K_M.gguf"
texts = []
texts.append("This is a test document.")
texts.append("This is a new test document.")

query_result1 = gpt4all_embd.embed_query(texts[0])
query_result2 = gpt4all_embd.embed_query(texts[1])
print(np.dot(query_result1, query_result2))

# ----------------------------------------------------------------------
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All

template = """Question: {question}
Answer: """ #Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
print(f"{prompt=}")

local_path = (
    # "./models/ggml-gpt4all-l13b-snoozy.bin"  # replace with your desired local file path
    "/Users/erlebach/data/llm_models/zephyr-7b-beta.Q4_K_M.gguf"  # replace with your desired local file path
)


# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Verbose is required to pass to the callback manager
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

# If you want to use a custom model add the backend parameter
# Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
#llm = GPT4All(model=local_path, backend="gptj", verbose=True)
#llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True)

# llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

# llm_chain.run(question)

reply = llm.generate(
    [question],
    max_length=100,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    repetition_penalty=1.0,
    num_beams=1,
    no_repeat_ngram_size=0,
    early_stopping=False,
    use_cache=True,
    model_specific_kwargs=None,
    streaming=True,
)

question = "The capital of France is, against any"
reply = llm.generate([question], max_tokens=1, streaming=False, temperature=0.9)
for i, token in enumerate(reply):
    print(f"{i},   {token=}")


