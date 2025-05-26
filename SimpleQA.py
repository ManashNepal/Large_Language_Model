from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

subject_content = """
In physics, gravity (from Latin gravitas 'weight'), also known as gravitation or a gravitational interaction, is a fundamental interaction, a mutual attraction between 
all massive particles. On Earth, gravity takes a slightly different meaning: the observed force between objects and the Earth. This force is dominated by the combined 
gravitational interactions of particles but also includes effect of the Earth's rotation. Gravity gives weight to physical objects and is essential to understanding 
the mechanisms responsible for surface water waves and lunar tides. Gravity also has many important biological functions, helping to guide the growth of plants through 
the process of gravitropism and influencing the circulation of fluids in multicellular organisms.
"""

question = input("Ask question: ")

prompt = (
    "Use the below information to answer the question\n"
    f"Subject Content : {subject_content}\n"
    f"Question : {question}\n"
    "Answer :"
)

input_ids = tokenizer.encode(prompt, return_tensors = "pt")

output = model.generate(
    input_ids,
    max_length = 200,
    num_return_sequences = 1,
    top_k = 50,
    top_p = 0.95,
    temperature = 0.8
)

generated_text = tokenizer.decode(output[0], skip_special_tokens = True).replace(prompt, "").replace("\n","").strip()

print(f"Question : {question}\n")
print(f"Answer : {generated_text}")