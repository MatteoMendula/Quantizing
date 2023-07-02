
from torchvision.io.image import read_image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import time

img = read_image("../000000024309.jpg")

# Step 1: Initialize model with the best available weights
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights, box_score_thresh=0.9)
model.cuda()
model.eval()

#print(img)
# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img).cuda()]

# Step 4: Use the model and visualize the prediction
start_time = time.time()
prediction = model(batch)
print("--- %s seconds ---" % (time.time() - start_time))
print(prediction)

