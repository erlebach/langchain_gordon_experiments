# Zephyr
I am running the pretrained version of Zephyr on my CPU on the Mac (not the GPU). Here are first results.
What is amazing is how close the structure of these sentences are to one another.
The code took about 102 seconds!!!

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