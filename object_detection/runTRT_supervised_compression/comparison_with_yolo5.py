# Import needed libraries and define the evaluate function
import pycuda.driver as cuda
import pycuda.autoinit
import time 
import tensorrt as trt
import numpy as np
import argparse 

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
latency_inf = []

# 640x427 imagesize
# Sample images from the COCO validation set
import torchvision.transforms as transforms
from PIL import Image
import torch
transform = transforms.Compose([
    transforms.ToTensor()
])
image = Image.open("car800x1280.jpg")
image = Image.open("car320x240.jpg")

image = Image.open("kitchen.jpg")

image = transform(image)
input = torch.tensor(image, dtype=torch.float32).cuda()

np.save("original_image.npy", input.cpu())

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
np.copyto(host_inputs[0], input.unsqueeze(0).ravel().cpu())

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
    latency_inf += [time.time()-start_time]
    time.sleep(0.113)
    # head_output_q = quantization_utils.quantize_tensor(host_outputs[0])


mean = sum(latency_inf) / len(latency_inf)
variance = sum([((x - mean) ** 2) for x in latency_inf]) / len(latency_inf)
res = variance ** 0.5
print("Mean: " + str(mean))
print("Variance: " + str(variance))
print("Standard deviation: " + str(res))

print("host_outputs[0]", host_outputs[0])
print("host_outputs[1]", host_outputs[1])
print("host_outputs[2]", host_outputs[2])
reshaped = host_outputs[0].reshape(1, 1, 104, 154)
print("reshaped dtype", reshaped.dtype)
print("reshaped type", type(reshaped))
np.save("reshaped.npy", reshaped)
print("--------- done ---------")

# host_outputs[0] [243. 243. 237. ...   7.   3.   0.]
# host_outputs[1] [0.4033]
# host_outputs[2] [121.7]


# from SSD_creator import SSD_creator
# from model import SSD300Tail
# import torch
# ssd_creator = SSD_creator("fp32", "IMAGENET1K_V2")
# tail = SSD300Tail(ssd_creator.ssd300.eval())
# tail.eval().cuda()

# head_output = quantization_utils.dequantize_tensor(torch.tensor(host_outputs[0].reshape(1, 256, 75, 75)), torch.tensor(host_outputs[1]), torch.tensor(host_outputs[2]))
# splitted_model_output = tail(head_output.cuda())

# results_per_input = utils.decode_results((splitted_model_output[0], splitted_model_output[1]))
# best_results_per_input_trt = [utils.pick_best(results, 0.40) for results in results_per_input]
# # Visualize results bare TensorRT
# utils.plot_results(best_results_per_input_trt, inputs, classes_to_labels)
