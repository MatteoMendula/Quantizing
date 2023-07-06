import sys
import cv2
import imutils
import time
import imutils
from PIL import Image
import numpy as np
from yoloDet import YoloTRT

# use path for library and engine file
model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.7, yolo_ver="v5")

pil_image = Image.open('../../../images/kitti_1.png')
numpy_array = np.array(pil_image)
resized_array = numpy_array
# resized_array = imutils.resize(numpy_array, width=600)
latencies = []

detections, unprocessed_results, t = model.Inference(resized_array)
print("unprocessed_results: {}".format(unprocessed_results))
image_from_array = Image.fromarray(resized_array)
image_from_array.save("yolov5.png")