import sys
import cv2
import imutils
import time
import imutils
from PIL import Image
import numpy as np
from yoloDet import YoloTRT

# use path for library and engine file
model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

pil_image = Image.open('../../../images/kitchen.jpg')
numpy_array = np.array(pil_image)
resized_array = numpy_array
# resized_array = imutils.resize(numpy_array, width=600)
latencies = []

WARMUPS = 50
INFERENCES = 1000

print("Warming up...")
for i in range(WARMUPS):
    detections, t = model.Inference(resized_array)

print("Starting inference...")
for i in range(INFERENCES):
    start_time = time.time()
    detections, t = model.Inference(resized_array)
    latencies += [time.time() - start_time]

# mean
print("Mean latency: ", sum(latencies) / len(latencies))
# variance
print("Variance: ", sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies))
# standard deviation
print("Standard deviation: ", (sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies)) ** 0.5)