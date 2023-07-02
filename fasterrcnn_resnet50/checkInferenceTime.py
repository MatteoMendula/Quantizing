
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import time
import torch
from io import BytesIO # "import StringIO" directly in python2
from PIL import Image
import os

def get_compressed_image(image_path, quality=40):
    from io import BytesIO # "import StringIO" directly in python2
    from PIL import Image
    import os
    im1 = Image.open(image_path)

    # here, we create an empty string buffer    
    buffer = BytesIO()
    im1.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    image = Image.open(buffer)
    return image

WARM_UP = 100
INFERENCES = 1000

img_path = "../images/kitchen.jpg"
jpeg_compression = 10

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.cuda()
model.eval()

# jpeg_compression image
img = get_compressed_image(img_path, jpeg_compression)

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img).cuda()]

latency = []

print("Warming up for " + str(WARM_UP) + " iterations.")
for i in range(WARM_UP):
    model(batch)

print("Running for " + str(INFERENCES) + " iterations.")
for i in range(INFERENCES):
    start_time = time.time()
    model(batch)
    latency.append(time.time() - start_time)

mean = sum(latency) / len(latency)
variance = sum([((x - mean) ** 2) for x in latency]) / len(latency)
res = variance ** 0.5
print("Mean: " + str(mean))
print("Variance: " + str(variance))
print("Standard deviation: " + str(res))


