
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import time
import torch
from io import BytesIO
from PIL import Image
import os
from flask import Flask, request, jsonify

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.cuda()
model.eval()
preprocess = weights.transforms()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image found in the request', 400
    
    image_file = request.files['image']
    image = Image.open(image_file)
    batch = [preprocess(image).cuda()]
    prediction = model(batch)

    my_prediction = {}
    my_prediction['detection'] = {}
    my_prediction['detection']["boxes"] = prediction[0]["boxes"].cpu().detach().numpy().tolist()
    my_prediction['detection']["labels"] = prediction[0]["labels"].cpu().detach().numpy().tolist()
    my_prediction['detection']["scores"] = prediction[0]["scores"].cpu().detach().numpy().tolist()

    return jsonify(my_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)



