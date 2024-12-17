import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rwkv7_model.configuration_rwkv7 import Rwkv7Config
from rwkv7_model.modeling_rwkv7 import Rwkv7ForCausalLM
from transformers import AutoConfig, AutoModel
AutoConfig.register("rwkv7", Rwkv7Config)
AutoModel.register(Rwkv7Config, Rwkv7ForCausalLM)
AutoModelForCausalLM.register(Rwkv7Config, Rwkv7ForCausalLM)

def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"""Instruction: {instruction}

Input: {input}

Response:"""
    else:
        return f"""User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: {instruction}

Assistant:"""


model = AutoModelForCausalLM.from_pretrained("../ckpt/RWKV7-Goose-0.1B-World3", trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("RWKV/v5-Eagle-7B-HF", trust_remote_code=True)

prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
#prompt = generate_prompt(text)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
output = model.generate(inputs["input_ids"], max_new_tokens=128, do_sample=False, top_k=0)
#output = model.generate(inputs["input_ids"], max_new_tokens=128, do_sample=True, temperature=1.0, top_p=0.7, top_k=0, )
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
