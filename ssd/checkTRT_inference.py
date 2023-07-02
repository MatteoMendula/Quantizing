# Import needed libraries and define the evaluate function
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import time 
import tensorrt as trt
import numpy as np
import argparse 
import inference_utils as utils

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
latency = 0

# 640x427 imagesize
# Sample images from the COCO validation set
uris = [
    './kitchen.jpg',
]

# create buffer
host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []
stream = cuda.Stream()

classes_to_labels= utils.get_coco_object_dictionary()

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
runtime = trt.Runtime(TRT_LOGGER)
engine_path = "{}/ssd_{}.trt".format(location_trt_folder, precision)

with open(engine_path, 'rb') as f:
    serialized_engine = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()


for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    host_mem = cuda.pagelocked_empty(size, np.float32)
    if precision == 'fp16':
        host_mem = cuda.pagelocked_empty(size, np.float16)
    cuda_mem = cuda.mem_alloc(host_mem.nbytes)

    bindings.append(int(cuda_mem))
    if engine.binding_is_input(binding):
        host_inputs.append(host_mem)
        cuda_inputs.append(cuda_mem)
    else:
        host_outputs.append(host_mem)
        cuda_outputs.append(cuda_mem)

start_preprocess = time.time()
# Format images to comply with the network input
inputs = [utils.prepare_input(uri) for uri in uris]
tensor = utils.prepare_tensor(inputs)
np.copyto(host_inputs[0], tensor[0].unsqueeze(0).ravel().cpu())
print("Preprocess time: ", time.time()-start_preprocess)

print("Startin warmup")
for _ in range(N_WARMUPS):
    start_time = time.time()
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
print("Starting inference test")
for _ in range(N_RUNS):
    start_time = time.time()
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
    latency += time.time()-start_time


print("Latency avg: ", latency/N_RUNS)
print(host_outputs[0].dtype)

results_per_input = utils.decode_results((torch.from_numpy(host_outputs[0].reshape(1, 4, 8732)), torch.from_numpy(host_outputs[1].reshape(1, 81, 8732))))
best_results_per_input_trt = [utils.pick_best(results, 0.40) for results in results_per_input]
# Visualize results bare TensorRT
utils.plot_results(best_results_per_input_trt, inputs, classes_to_labels)
