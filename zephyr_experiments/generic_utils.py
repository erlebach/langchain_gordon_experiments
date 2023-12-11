from lib2to3.pgen2 import grammar
import requests
import numpy as np
import re
import json
import jinja2
import yaml
from typing import Tuple, Any

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
        "max_tokens": 200,
        "messages": messages,  # Required for chat
        "temperature": temperature,
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

    def print(self):
        for d in self.messages:
            print(f"{d['role']}: {d['content']}")


class MistralMessages(Messages):
    def __init__(self, subject, additional_context):
        super().__init__()
        self.additional_context = additional_context
        self.subject = subject

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


class Conversation:
    def __init__(
        self,
        instructions,
        subject: str,
        authorA: str,
        authorB: str,
        llmA,
        llmB,
        msgsA: Messages,
        msgsB: Messages,
        output_format=None,
        grammar=None,
    ):
        self.instructions = instructions
        self.subject = subject
        self.authorA = authorA
        self.authorB = authorB
        self.llmA = llmA
        self.llmB = llmB
        self.msgsA = msgsA
        self.msgsB = msgsB
        self.grammar = grammar
        self.output_format = output_format

        # messages beyond context
        self.stringA = ""
        self.stringB = ""

        # Replies by model
        self.conversation: list[dict] = []

        self.initiate_conversation()

    def initiate_conversation(self):
        """
        Initiate the conversation with a question.
        """
        self.conversation.append("")
        self.conversation.append(
            f"{self.authorA}: Hello {self.authorB}! Let us have a discussion on {self.subject}. Get the ball rolling!"
        )

    def update_strings(self, dct):
        msg = dct["Interlocutor"] + ": " + dct["Reply"]
        # print("msg: ", msg)
        self.stringA += "\n" + msg
        self.stringB += "\n" + msg

    def update_msgs(self, author, msg):
        """
        Add a new message to the conversation, defined by a list of dictionaries.
        """
        self.msgsA.add(author, msg)
        self.msgsB.add(author, msg)

    def get_reply(self, prompt):
        prompt = re.sub(" {2,}", "", prompt)

        """
        try:
            print(f"\n==> conversation[-2]: {self.conversation[-2]}\n...")
        except:
            pass

        try:
            print(f"\n==> conversation[-1]: {self.conversation[-1]}\n...")
        except:
            pass
        """

        if self.grammar is not None:
            reply = self.llmA(prompt)
        else:
            reply = self.llmA(prompt)
            raise "ErrorA"

        try:
            # print(f"\nBEFORE, reply: --{reply}--\n")
            if reply[-1] != "}":
                reply.append("}")
            # reply = re.sub("\n", " ", reply)
            # split accordint white space (including newlines)
            reply = " ".join(reply.split())
            # print("=====")
            # print(f"\nAFTER, reply: --{reply}--\n")
            dct = json.loads(reply)  # ERROR! WHY IS THAT?
        except:
            print("ERROR")

        # self.update_strings(dct)
        content = dct["Reply"]
        return content

    def multi_turn(self, nb_turns):
        for turn in range(nb_turns):
            print("\n===========================================")
            print(f">>>> turn # {turn}")
            self.single_turn()

    def get_prompt(
        self,
        author_speaking,
        author_replied_to,
    ):
        prompt = (
            self.instructions
            + self.subject
            + f"\n{author_speaking}: {self.conversation[-1]}\n{author_speaking}: \
            [INST]]Reply to {author_replied_to}.[/INST]"
        )
        print(f"\nprompt: {prompt}")
        return prompt

    def single_turn(self):
        prompt = self.get_prompt(self.authorB, self.authorA)
        content = self.get_reply(prompt)
        reply = f"{self.authorB}: {content}"
        self.conversation.append(reply)
        print(f"\n{reply}")

        prompt = self.get_prompt(self.authorA, self.authorB)
        content = self.get_reply(prompt)
        reply = f"{self.authorA}: {content}"
        self.conversation.append(reply)
        print(f"\n{reply}")

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


def apply_templates(filenm: str) -> Tuple[dict[str, Any], dict[str, str]]:
    """
    Apply templates to the given file.

    Args:
        filenm (str): The path to the file containing the templates.

    Returns: tuple
        dict: yaml representation of the file as a dictionary
        dict: A dictionary containing the rendered texts for each template key
    """
    env = jinja2.Environment()
    with open(filenm, "r") as f:
        data = yaml.safe_load(f)

    rendered_texts = {}
    for key, template_text in data["conversation"].items():
        template = env.from_string(template_text)
        rendered_texts[key] = template.render(data)
    return data, rendered_texts


# ----------------------------------------------------------------------
