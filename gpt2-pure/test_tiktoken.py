import tiktoken

text = "hello, how are you?"
enc = tiktoken.get_encoding("gpt2")
encoded_input = enc.encode(text)
print("encoded_input:", encoded_input)
decode_output = enc.decode(encoded_input)
print("decode_output:", decode_output)
assert decode_output == text

from load_gpt2_model import load_gpt2_model

config_json = 'openai-gpt2/config.json'
pytorch_model_bin = 'openai-gpt2/pytorch_model.bin'
gpt2 = load_gpt2_model(config_json, pytorch_model_bin)

from generate import generate_greedy
import torch

idx = (torch.tensor(encoded_input, dtype=torch.long)[None, ...])
output = generate_greedy(gpt2, idx, max_new_tokens=50)
gpt2_output = enc.decode(output[0].tolist())
print("gpt2_output:")
print(gpt2_output)
