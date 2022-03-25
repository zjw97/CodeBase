import torch
import onnxruntime as ort

def torch2onnx(model,
               dummy_input,
               save_path,
               verbose=False,
               input_names=["input"],
               output_names=["output"]):
    torch.onnx.export(model, dummy_input, save_path,
                      verbose=verbose, input_names=input_names,
                      output_names=output_names)
    print("torch to onnx finished!")

def runtime(onnx_path):
    session = ort.InferenceSession(onnx_path)





