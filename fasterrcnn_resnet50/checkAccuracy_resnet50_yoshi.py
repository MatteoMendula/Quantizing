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
warnings.filterwarnings('ignore')


use_cuda = True
override_prev_results = False

import itertools
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor()
])
def evaluate_coco(img_path, set_name, image_ids, coco, model, weights, threshold=0.05):
    results = []

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']
        #print(image_path)
        #weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        img = Image.open(image_path)

        image = transform(img)
        my_image = torch.unsqueeze(torch.Tensor(image), 0)
        input = torch.tensor(my_image, dtype=torch.float32)
        x = input.cuda()
        #x[0].cuda()
        #print(x.size())
        #features, regression, classification, anchors = model(x)
        out_head, other_info = model[0](x)
        pred = model[1](*(*out_head, *other_info))[1]

        scores = pred[0]['scores']
        class_ids = pred[0]['labels']
        rois = pred[0]['boxes']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                #print(image_result)
                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f"{set_name}_bbox_results.json"

    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

def _eval(coco_gt, image_ids, pred_json_path):
    cat_num=[1,2,3,4,5,6,7,9,16,17,18,19,20,21,44,62,63,64,67,72]
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    for e in cat_num:
        print("----------------------------------------------")
        print("Category "+ str(e))
        print("----------------------------------------------")
        coco_eval.params.catIds = [e]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    print("----------------------------------------------")
    print("Overall")
    print("----------------------------------------------")
    E = COCOeval(coco_gt, coco_pred, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f}".format(E.stats[0]))


if __name__ == '__main__':
    path_to_annotations="../coco/"
    coco_annotation_file_path = path_to_annotations+"annotations/instances_val2017.json"
    VAL_GT = path_to_annotations+"annotations/instances_val2017.json"
    VAL_IMGS = path_to_annotations+"val2017/"
    MAX_IMAGES = 50000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    SET_NAME="yoshi"
    
    if override_prev_results or not os.path.exists(f"{SET_NAME}_bbox_results.json"):
        # Step 1: Initialize model with the best available weights
        #weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        #model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT

        head_model_loaded = torch.jit.load('./models/head_model.pt').eval().cuda()
        tail_model_loaded = torch.jit.load('./models/tail_model.pt').eval().cuda()

        model = (head_model_loaded, tail_model_loaded)

        evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model, weights)

    _eval(coco_gt, image_ids, "{}_bbox_results.json".format(SET_NAME))
