from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"

model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model.eval() # setting the model to evaluation mode

prompt = input("Enter the prompt: ")

input_ids = tokenizer.encode(prompt, return_tensors = "pt")

output = model.generate(input_ids, max_new_tokens = 10, top_k = 100)

predicted_text = tokenizer.decode(output[0])

print(f"The predicted text : {predicted_text}")


