import torch
from transformers import BertTokenizer, BertModel, BertConfig
import json

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')["input_ids"]
with open ("word_ids.json", "w") as f:
	f.write(json.dumps(encoded_input.tolist()))
output = model(encoded_input)["last_hidden_state"]

print (output)

input_names = ["input_ids"]
output_names = ["output"]

torch.onnx.export(model,
    encoded_input,
    "model.onnx",
    export_params=True,
    do_constant_folding=True,
    input_names = input_names,
    output_names = output_names,
    opset_version=10, 
    dynamic_axes = {"input_ids": {1: 'sent length'}})

