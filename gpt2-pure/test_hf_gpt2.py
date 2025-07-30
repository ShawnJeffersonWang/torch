from transformers import GPT2LMHeadModel

hf_model = GPT2LMHeadModel.from_pretrained('openai-gpt2')

from transformers import GPT2Tokenizer

text = "hello, how are you?"
tokenizer = GPT2Tokenizer.from_pretrained('openai-gpt2')
encoded_input = tokenizer.encode(text, return_tensors='pt')
print("input text:", text)
print("input idx:", encoded_input)

output = hf_model.generate(encoded_input, max_length=50)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)
