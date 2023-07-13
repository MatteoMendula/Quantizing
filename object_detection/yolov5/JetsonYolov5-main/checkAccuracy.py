from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,FasterRCNN_MobileNet_V3_Large_FPN_Weights,fasterrcnn_mobilenet_v3_large_fpn
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from typing import Union
import warnings
import os
import json

import sys
import cv2
import imutils
import time
import imutils
from PIL import Image
import numpy as np
from yoloDet import YoloTRT
import torch

warnings.filterwarnings('ignore')

use_cuda = True
override_prev_results = True

import numpy as np
from io import BytesIO # "import StringIO" directly in python2
from PIL import Image
import os

def parse_label(label):
    _label = label + 1
    if _label < 12:
        return _label
    
    if 12 <= _label and _label < 25:
        return _label + 1
    elif 25 <= _label and _label < 27:
        return _label + 2
    elif 27 <= _label and _label < 41:
        return _label + 4
    elif 41 <= _label and _label < 61:
        return _label + 5
    elif 61 == _label:
        return _label + 6
    elif 62 == _label:
        return _label + 8
    elif 63 <= _label and _label < 74:
        return _label + 9
    elif 74 <= _label:
        return _label + 10

def evaluate_coco(img_path, set_name, image_ids, coco, model, weights):
    results = []
    classes = set()
    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        image = Image.open(image_path)
        x = np.array(image)
        if len(x.shape) != 3:
            print(x.shape)
            continue

        pred = model.Inference(x)
        # image_from_array = Image.fromarray(x)
        # image_from_array.save("test_acc.png")
        # time.sleep(1)

        scores = torch.tensor(pred[1]['scores']).cuda()
        class_ids = torch.tensor(pred[1]['labels']).cuda()
        rois = torch.tensor(pred[1]['boxes']).cuda()

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                # make it compliant with coco 91 classes
                label = parse_label(label)
                box = rois[roi_id, :]
                classes.add(label)
                image_result = {
                    'image_id': image_id,
                    'category_id': label,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f"{set_name}_bbox_results.json"

    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

    print("len results: {}".format(len(results)))
    print("classes: {}".format(classes))
    time.sleep(5)

def _eval(coco_gt, image_ids, pred_json_path):
    cat_num=[1,2,3,4,5,6,7,9,16,17,18,19,20,21,44,62,63,64,67,72]
    coco_pred = coco_gt.loadRes(pred_json_path)
    print("----------------------------------------------")
    print("Overall")
    print("----------------------------------------------")
    E = COCOeval(coco_gt, coco_pred, iouType='bbox')
    E.params.imgIds = image_ids
    # E.params.catIds = cat_num
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f}".format(E.stats[0]))


if __name__ == '__main__':
    path_to_annotations="../../../../../coco2017/"
    coco_annotation_file_path = path_to_annotations+"annotations/instances_val2017.json"
    VAL_GT = path_to_annotations+"annotations/instances_val2017.json"
    VAL_IMGS = path_to_annotations+"val2017/"
    MAX_IMAGES = 50000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    SET_NAME="yolov5"
    if override_prev_results or not os.path.exists(f"{SET_NAME}_bbox_results.json"):
        model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        
        evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model, weights)

    _eval(coco_gt, image_ids, "{}_bbox_results.json".format(SET_NAME))
