from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.chains import LLMChain

template = """
Translate the follwing English sentence to French

"{english_sentence}"
"""

prompt = PromptTemplate(template=template, input_variables=["english_sentence"])


model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

hf_pipeline = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens = 100, temperature = 0.3)

llm = HuggingFacePipeline(pipeline = hf_pipeline)

translator_chain = LLMChain(llm = llm, prompt = prompt)

english_input = input("Give me a sentence in English: ")

translated_text = translator_chain.invoke(english_input)

print(translated_text)

