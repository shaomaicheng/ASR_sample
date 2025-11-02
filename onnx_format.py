import onnxruntime as ort

sess = ort.InferenceSession("model/model.onnx")

# 输入
print("model.onnx输入")
for input in sess.get_inputs():
    print(f"{input.name}: shape={input.shape}, type={input.type}")

print("mode.onnx输出：")
for output in sess.get_outputs():
    print(f"{output.name}: shape={output.shape}, type={output.type}")



sess_decoder = ort.InferenceSession("model/decoder.onnx")

# 输入
print("decoder.onnx输入")
for input in sess_decoder.get_inputs():
    print(f"{input.name}: shape={input.shape}, type={input.type}")

print("decoder.onnx输出：")
for output in sess_decoder.get_outputs():
    print(f"{output.name}: shape={output.shape}, type={output.type}")


sess_quant = ort.InferenceSession("model/model_quant.onnx")

# 输入
print("mode_quant.onnx输入")
for input in sess_quant.get_inputs():
    print(f"{input.name}: shape={input.shape}, type={input.type}")

print("mode_quant.onnx输出：")
for output in sess_quant.get_outputs():
    print(f"{output.name}: shape={output.shape}, type={output.type}")