import torch
from transformers import AutoTokenizer, BloomForCausalLM

model_dir = "/Users/wangyan/Downloads/bloom-560m"
print(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = BloomForCausalLM.from_pretrained(model_dir)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits
print(logits)