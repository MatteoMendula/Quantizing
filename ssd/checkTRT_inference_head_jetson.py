# Import needed libraries and define the evaluate function
import pycuda.driver as cuda
import pycuda.autoinit
import time 
import tensorrt as trt
import numpy as np
import argparse 
import inference_utils as utils
import quantization_utils as quantization_utils 

parser = argparse.ArgumentParser(description='Parser for SSD300')
parser.add_argument('-p', '--precision', default="fp32", type=str)
parser.add_argument('-l', '--location_trt_folder', default=".", type=str)
args = vars(parser.parse_args())

precision = args['precision']
location_trt_folder = args['location_trt_folder']

from matplotlib import pyplot as plt
import matplotlib.patches as patches

N_WARMUPS = 50
N_RUNS = 1000
latency_inf = 0

# 640x427 imagesize
# Sample images from the COCO validation set
uris = [
    './kitchen.jpg',
]

classes_to_labels= utils.get_coco_object_dictionary()

# Format images to comply with the network input
inputs = [utils.prepare_input(uri) for uri in uris]
# parsing the size of the input tensor to fp12/32 inside the prepare_tensor function
tensor = utils.prepare_tensor(inputs, should_half=precision=='fp16')
print("tensor type: ", tensor[0].dtype)
print("[{}]original size: ".format(precision), tensor[0].element_size() * tensor[0].nelement())

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
runtime = trt.Runtime(TRT_LOGGER)
engine_path = "{}/head_{}.trt".format(location_trt_folder, precision)

with open(engine_path, 'rb') as f:
    serialized_engine = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)
    
# create buffer
host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []
stream = cuda.Stream()

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    if precision == 'fp16':
        print("precision is fp16")
        host_mem = cuda.pagelocked_empty(size, np.float16)
    else:
        host_mem = cuda.pagelocked_empty(size, np.float32)
    cuda_mem = cuda.mem_alloc(host_mem.nbytes)

    bindings.append(int(cuda_mem))
    if engine.binding_is_input(binding):
        host_inputs.append(host_mem)
        cuda_inputs.append(cuda_mem)
    else:
        # overriding host_mem with tensor memory to int8
        host_outputs.append(host_mem)
        cuda_outputs.append(cuda_mem)
context = engine.create_execution_context()
np.copyto(host_inputs[0], tensor[0].unsqueeze(0).ravel().cpu())

print("Starting warmup")
for _ in range(N_WARMUPS):
    start_time = time.time()
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
    cuda.memcpy_dtoh_async(host_outputs[2], cuda_outputs[2], stream)
    stream.synchronize()
print("Starting inference test")
for _ in range(N_RUNS):
    start_time = time.time()
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
    cuda.memcpy_dtoh_async(host_outputs[2], cuda_outputs[2], stream)
    latency_inf += time.time()-start_time


print("Latency avg: ", latency_inf/N_RUNS)
print("host_outputs[0]", host_outputs[0])
print("host_outputs[1]", host_outputs[1])
print("host_outputs[2]", host_outputs[2])
reshaped = host_outputs[0].reshape(1, 256, 75, 75)
print("reshaped", reshaped.dtype)
print("--------- done ---------")

