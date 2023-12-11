import re
import json

reply = '{  "Reply": "Dear Professor Smolin, \
I appreciate your point of view on the importance of both loop quantum gravity and string theory. Both of these approaches hold promise in addressing some of the major challenges in theoretical physics, such as reconciling general relativity with quantum mechanics. \
 \
However, I also think that there are significant differences between loop quantum gravity and string theory that make them better suited for different areas of research. For instance, string theory is a candidate for a theory of everything, while loop quantum gravity focuses more on the behavior of spacetime at small scales. \
 \
I agree that continued collaboration among physicists with expertise in both areas will be essential in order to fully explore the possibilities and challenges presented by these theories. It may even be possible to develop new approaches that combine elements from both, or build on their shared features to develop a more complete understanding of the universe." \
}'

reply = re.sub("\n", " ", reply)

print(f"\nAFTER, reply: --{reply}--\n")

dct = json.loads(reply)

print(dct['Reply'])
