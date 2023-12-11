import utils as u

msgs = u.MistralMessages()
# <s>[INST]{prompt}[/INST]</s>
msgs.add_instruction("system", "This is an instruction")
msgs.add("user", "gordon")
print(msgs())

messages = msgs()
print(messages)

all_content = []

for msg in messages:
    for k, v in msg.items():
        print(f"==> {k}: {v}")
    all_content.append(v)

print("============================")
print(all_content)
