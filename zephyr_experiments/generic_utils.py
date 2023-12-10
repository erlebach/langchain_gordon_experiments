from lib2to3.pgen2 import grammar
import requests
import numpy as np
import re
import json

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
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        print(f"An error occurred: {e}")
        return None

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

    def print(self):  # , role):
        # print(f"{role}:")
        for d in self.messages:
            print(f"{d['role']}: {d['content']}")


class MistralMessages(Messages):
    def __init__(self):
        super().__init__()

    def add(self, role, content):
        strg_dict = dict(role=role + ":", content=f"{content}")
        self.messages.append(strg_dict)

    def add_instruction(self, role, content):
        strg_dict = dict(role="", content=f"<s>[INST]{content}[/INST]</s>")
        self.messages.append(strg_dict)

    def full_context(self):
        all_content = ""
        for msg in self.messages:
            all_content += f"\n{msg['role']} {msg['content']}"
        return all_content.strip()  # remove leading and trailing spaces


# # Not clear this class does anything
# class MistralGeneric(Messages):
#     def __init__(self):
#         super().__init__()

#     def add(self, role, content):
#         strg_dict = dict(role=role + ":", content=f"[INST]{content}[/INST]")
#         self.messages.append(strg_dict)

#     def add_instruction(self, role, content):
#         strg_dict = dict(role="", content=f"<s>[INST]{content}[/INST]</s>")
#         self.messages.append(strg_dict)

#     def full_context(self):
#         # Inefficient to reconstruct the full context each time.
#         # More efficient to add new messages to the context, which is a list.
#         # self.messages: is a list of dictionaries
#         all_content = ""
#         for msg in self.messages:
#             # The colon between role and content is included in the dictionary
#             # Why? Because the instruction has no role. The structure is different.
#             all_content += f"\n{msg['role']} {msg['content']}"
#         return all_content


class Conversation:
    def __init__(self, authorA, authorB, llmA, llmB, msgsA, msgsB, grammar=None):
        self.authorA = authorA
        self.authorB = authorB
        self.llmA = llmA
        self.llmB = llmB
        self.msgsA = msgsA
        self.msgsB = msgsB
        self.grammar = grammar

        # messages beyond context
        self.stringA = ""
        self.stringB = ""

    def update_strings(self, dct):
        msg = dct["Interlocutor"] + ": " + dct["Reply"]
        # print("msg: ", msg)
        self.stringA += "\n" + msg
        self.stringB += "\n" + msg
        # print("self.stringA: ", self.stringA)
        # print("self.stringB: ", self.stringB)

    def update_msgs(self, author, msg):
        """
        Add a new message to the conversation, defined by a list of dictionaries.
        """
        self.msgsA.add(author, msg)
        self.msgsB.add(author, msg)
        # self.msgsA.add(f"{self.authorA}", msg_authorA)
        # self.msgsB.add(f"{self.authorA}", msg_authorA)

    def multi_turn(self, nb_turns):
        for turn in range(nb_turns):
            print("\n===========================================")
            print(f">>>> turn # {turn}")
            self.single_turn()

    def single_turn(self):
        # Add both conversations in the same buffer. This assumes that both speakers
        # have perfect memory. Ultimately, that is not the case. I should create a different
        # memory buffer for each speaker with different properties. For example, 5 speakers
        # could be in a conversation, but only conversations within a certain range can be
        # heard by a given person. If the people were Borg, each Borg would hear everything.

        # Set up reply by authorA
        contextA = self.msgsA.full_context()
        print("==> contextA: ", contextA)
        print("==> self.stringA= ", self.stringA)

        # print("\n==> before first question, msgsA: ", self.msgsA())
        if self.grammar is not None:
            msg_authorA = self.llmA(
                contextA + self.stringA, grammar=self.grammar
            )  # the argument must be a string for Mistral
        else:
            msg_authorA = self.llmA(contextA)
            raise "ERROR"

        print("\n==> AuthorA reply: ", msg_authorA)  # Empty string!!! WHY?

        # Process json string
        dct = json.loads(msg_authorA)[0]
        self.update_strings(dct)

        # Why does self.llmA return no answer?
        # msg_authorA = self.llmA("do you like flying into space?")
        # self.update_msgs(self.authorA, msg_authorA)

        contextB = self.msgsB.full_context()
        print("==> contextB: ", contextB)
        print("==> self.stringB= ", self.stringB)

        # Set up reply by authorB
        # contextB = self.msgsB.full_context()
        if self.grammar is not None:
            print(f"\n{contextB+self.stringB=}")
            msg_authorB = self.llmB(
                contextB + self.stringB, grammar=self.grammar
            )  # the argument must be a string for Mistral
        else:
            msg_authorB = self.llmB(contextB)
            raise "error"

        if msg_authorB == "":
            print("error: msg_authorB is empty string")
            raise "error"

        # print("contextB: ", contextB)  # should contain the first reply
        print(f"\n==> AuthorB reply: {msg_authorB=}")

        dct = json.loads(msg_authorB)[0]
        # print(dct["Interlocutor"] + ": ", dct["Reply"])
        self.update_strings(dct)

        # message as a string
        # msg = dct["Interlocutor"] + ": ", dct["Reply"]
        # mgsA.add("User", msg)

        print("\n\n\n======================\n\n\n")
        # raise "error"

    def print_both_contexts(self):
        print(f"\n\n====> print msgsA, {self.authorA}'s context")
        self.msgsA.print()
        print(f"\n\n====> print msgsB, {self.authorB}'s context")
        self.msgsB.print()


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
