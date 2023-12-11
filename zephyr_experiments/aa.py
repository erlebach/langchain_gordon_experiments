import re

msg = '  {"m": "d   \
 \
       e"}  '

#print("gordon")
msg = msg.replace('\n', ' ')
print(f"{msg=}")
msg = re.sub(' +', ' ', msg)
print(f"{msg=}")
print(msg['m'])
quit()

reply = re.sub(r" +", " ", reply)
reply = re.sub(r'{ *"', '{"', reply)
reply = re.sub(r'" *}', '"}', reply)
print("reply: ", reply)


#dct = json.loads(reply)  # ERROR! WHY IS THAT?

for m in msg:
    if m == '\n':
        print("return")

