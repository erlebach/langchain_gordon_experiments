import requests
import numpy as np
from pprint import pprint
import re

api_key = "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"


def run_model(model, url, payload):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }

    response = requests.post(url, json=payload, headers=headers)

    try:
        if response.status_code != 200:
            print(f"{model}, Status code: {response.status_code}")
    except:
        return null

    # print("response: ", response)
    # print("response.text: ", response.text)
    # response = response["choices"][0]["message"]["content"]
    response = response.json()["choices"][0]["message"]["content"]
    # response = eval(response.text)["choices"][0]["message"]["content"]
    response = re.sub("\n", "", response)
    response = response.split("}", 1)[0] + "}"

    # print("response: ", response)
    return response


# ----------------------------------------------------------------------
def create_payload(model, messages, temperature):
    payload = {
        "model": model,
        # "prompt": "How many stars are there in the Milky-Way?",  # Required for text
        "max_tokens": 200,
        "messages": messages,  # Required for chat
        "temperature": temperature,
        # "top_p": 1,
        # "top_k": 10,
        # "stream": False,
        # "presence_penalty": 0.0,
        # "frequency_penalty": 0.1
    }
    return payload


class Messages:
    def __init__(self):
        self.messages = []

    def add(self, role, content):
        strg_dict = dict({"role": role, "content": content})
        self.messages.append(strg_dict)

    def __call__(self):
        return self.messages

    def print(self, role):
        print(f"\n\n{role}:")
        for d in self.messages:
            print(f"{d['role']}: {d['content']}")


class MistralMessages(Messages):
    def __init__(self):
        super().__init__()

    def add(self, role, content):
        strg_dict = dict(role=role, content=f"[INST]{content}[/INST]")
        self.messages.append(strg_dict)

    def add_instruction(self, role, content):
        strg_dict = dict(role=role, content=f"<s>[INST]{content}[/INST]</s>")
        self.messages.append(strg_dict)

    def full_context(self):
        all_content = ""
        for msg in self.messages:
            all_content += f"\n{msg['role']}: {msg['content']}"
        return all_content


# ----------------------------------------------------------------------
class ListIterator:
    def __init__(self, input_list):
        self.input_list = np.random.permutation(input_list)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.input_list):
            current_element = self.input_list[self.index]
            self.index += 1
            return current_element
        else:
            raise StopIteration


# ----------------------------------------------------------------------


def print_both_contexts(msgsA, msgsB):
    print("\n\n====> print msgsA, Newton's context")
    msgsA.print("")
    print("\n\n====> print msgsB, Einstein's context")
    msgsB.print("")
