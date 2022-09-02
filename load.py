import onnx
import onnxruntime

from transformers import BertTokenizer

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')["input_ids"]

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession("model.onnx")

ort_inputs = {ort_session.get_inputs()[i].name: to_numpy(x) for i, x in enumerate([encoded_input])}
ort_outs = ort_session.run(None, ort_inputs)

print (ort_outs)