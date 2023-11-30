# Author: G. Erlebacher
# The model is run on Hugginface, but not using Langchain

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

class HuggingFacePipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_model_id(cls, model_id, task, trust_remote_code=False, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote_code, use_flash_attention_2=False, **model_kwargs)
        return cls(model, tokenizer)

    def generate_text(self, prompt, temperature=0.8, **generation_kwargs):
        self.model.config.temperature=temperature
        self.model.config.do_sample=True
        gen_cfg = GenerationConfig.from_model_config(self.model.config)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, generation_config=gen_cfg)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

#----------------------------------------------------------------------
if __name__ == "__main__":
    # Usage
    llm = HuggingFacePipeline.from_model_id(
        #model_id="NousResearch/Yarn-Mistral-7b-64k",
        model_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        trust_remote_code=True,
        model_kwargs={"max_length": 50}
    )


    # Define a prompt for the model
    prompt = "Please explain how I could travel faster arbitrarily close to the speed of light?"

    temperature = 0.1
    for i in range(3):
        prompt = "Are you happy?"   # Cannot use a list
        text = llm.generate_text(prompt, temperature=temperature)
        print(text)
