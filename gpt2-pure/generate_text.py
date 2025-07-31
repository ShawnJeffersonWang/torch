from load_gpt2_model import load_gpt2_model

config_json = 'openai-gpt2/config.json'
pytorch_model_bin = 'openai-gpt2/pytorch_model.bin'
gpt2 = load_gpt2_model(config_json, pytorch_model_bin)

from transformers import GPT2Tokenizer

text = "calculate 1+2=? please"
tokenizer = GPT2Tokenizer.from_pretrained('openai-gpt2')
encoded_input = tokenizer.encode(text, return_tensors='pt')
print("input text:", text)
print("input idx:", encoded_input)

print("gpt2.generate:")
print("###1 begin:")
output = gpt2.generate(encoded_input, max_new_tokens=50)
gpt2_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(gpt2_output)
print("###1 end\n")

print("generate_random:")
print("###2 begin:")
from generate import generate_random

output2 = generate_random(gpt2, encoded_input, max_new_tokens=50)
gpt2_output2 = tokenizer.decode(output2[0], skip_special_tokens=True)
print(gpt2_output2)
print("###2 end\n")

print("generate_greedy:")
print("###3 begin:")
from generate import generate_greedy

output2 = generate_greedy(gpt2, encoded_input, max_new_tokens=50)
gpt2_output2 = tokenizer.decode(output2[0], skip_special_tokens=True)
print(gpt2_output2)
print("###3 end\n")
