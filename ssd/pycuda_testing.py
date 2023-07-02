import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.autoinit
import numpy

from quantization_utils import quantize_tensor_pycuda

def dequantize_tensor(q_x):
    return q_x[1] * (q_x[0] - q_x[2])

x = numpy.random.randn(4,4).astype(numpy.float32)
a_gpu = gpuarray.to_gpu(x)
q_x, scale, zero_point = quantize_tensor_pycuda(a_gpu)
print(q_x)
print(scale)
print(zero_point)

deq = dequantize_tensor((q_x, scale, zero_point))
print(sum(deq - x))
