from pytorch_quantization import tensor_quant
import torch
import ssd.utils.quantization_utils as quantization_utils

# Generate random input. With fixed seed 12345, x should be
# tensor([0.9817, 0.8796, 0.9921, 0.4611, 0.0832, 0.1784, 0.3674, 0.5676, 0.3376, 0.2119])
# torch.manual_seed(12345)
x = torch.rand(10)
print("original tensor", x)

# fake quantize tensor x. fake_quant_x will be
# tensor([0.9843, 0.8828, 0.9921, 0.4609, 0.0859, 0.1797, 0.3672, 0.5703, 0.3359, 0.2109])
# fake_quant_x = tensor_quant.fake_tensor_quant(x, x.abs().max())
# print(fake_quant_x)

# quantize tensor x. quant_x will be
# tensor([126., 113., 127.,  59.,  11.,  23.,  47.,  73.,  43.,  27.])
# with scale=128.0057
quant_x, scale = tensor_quant.tensor_quant(x, x.abs().max())
print(quant_x.dtype)
quant_x = quant_x.to(torch.int8)
print("quantized tensor", quant_x)
print(scale)

# dequantize quant_x
dequantized_x = quant_x.to(torch.float32) / scale
print("dequantized_x", dequantized_x)

quantized_matte = quantization_utils.quantize_tensor(x)
print("quantized_matte", quantized_matte)
dequantized_matte = quantization_utils.dequantize_tensor(quantized_matte)
print("dequantized_matte", dequantized_matte)

print("diff matte", sum(dequantized_matte - x))
print("diff torch", sum(dequantized_x - x))