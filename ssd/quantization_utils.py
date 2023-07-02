from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath

import pycuda.autoinit
import numpy

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def quantize_tensor_torch(x, cuda = True, num_bits=8):
    assert torch.is_tensor(x)
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val.item() - min_val.item()) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point.item()


    # zero_point = int(zero_point)
    zero_point_tensor = torch.full(x.shape,zero_point, dtype=x.dtype)
    scale_tensor = torch.full(x.shape,scale, dtype=x.dtype)
    if cuda:
        zero_point_tensor = zero_point_tensor.cuda()
        scale_tensor = scale_tensor.cuda()
        x = x.cuda()
    q_x = x.div(scale_tensor).add(zero_point_tensor)
    q_x.clamp_(qmin, qmax).round_()

    q_x = q_x.to(x.dtype)
    scale = torch.tensor(scale, dtype=x.dtype)
    zero_point = torch.tensor(zero_point, dtype=x.dtype)

    return (q_x, scale, zero_point)

def dequantize_tensor_pytorch(quantized_tensor, scale, zero_point, cuda = True):
    zero_point_tensor = torch.full(quantized_tensor.shape, zero_point.item(), dtype=quantized_tensor.dtype)
    scale_tensor = torch.full(quantized_tensor.shape, scale.item(), dtype=quantized_tensor.dtype)
    if cuda:
        zero_point_tensor = zero_point_tensor.cuda()
        scale_tensor = scale_tensor.cuda()
    return quantized_tensor.sub(zero_point_tensor).mul(scale_tensor)

def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    # if torch.is_tensor(q_x):
    #     q_x.clamp_(qmin, qmax).round_()
    #     q_x = q_x.round().byte()
    # else:
    #     # numpy
    #     q_x = np.clip(q_x, qmin, qmax).round()
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(t, s, z):
    return s * (t.float() - z)


def quantize_model(model):
    qparams = {}

    for n, p in model.state_dict().items():
        qp = quantize_tensor(p)
        qparams[n + '.quantization.scale'] = torch.FloatTensor([qp.scale])
        qparams[
            n + '.quantization.zero_point'] = torch.ByteTensor([qp.zero_point])
        p.copy_(qp.tensor)
    model.type('torch.ByteTensor')
    for n, p in qparams.items():
        model.register_buffer(n, p)
    model.quantized = True

    return qp.scale, qp.zero_point

def dequantize_model(model):
    model.float()
    params = model.state_dict()
    for n, p in params.items():
        if 'quantization' not in n:
            qp = QTensor(tensor=p,
                         scale=params[n + '.quantization.scale'][0],
                         zero_point=params[n + '.quantization.zero_point'][0])
            p.copy_(dequantize_tensor(qp))
            model.register_buffer(n + '.quantization.scale', None)
            model.register_buffer(n + '.quantization.zero_point', None)
    model.quantized = None

def simple_compressione(tensor):
    import gzip
    import pickle
    compressed_tensor = gzip.compress(pickle.dumps(tensor))
    compressed_size = len(compressed_tensor)
    print("Compressed size:", compressed_size)
    return compressed_tensor

def quantize_tensor_pycuda(x):
    qmin = 0
    qmax = 2**8 - 1.
    x_gpu = gpuarray.to_gpu_async(x)
    min_val, max_val = pycuda.gpuarray.min(x_gpu), pycuda.gpuarray.max(x_gpu)
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    # zero_point = int(zero_point.get())
    # zero_point_np_array = numpy.full(x.shape, zero_point).astype(numpy.int)
    # zero_point_gpu_array = gpuarray.to_gpu(zero_point_np_array)
    q_x = x_gpu.__div__(scale).__add__(zero_point) 
    q_x_int8 = cumath.ceil(q_x)
    return (q_x_int8, scale, zero_point)