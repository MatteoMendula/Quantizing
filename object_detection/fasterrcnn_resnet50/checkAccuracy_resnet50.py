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
override_prev_results = True

import numpy as np
from io import BytesIO # "import StringIO" directly in python2
from PIL import Image
import os

def evaluate_coco(img_path, set_name, image_ids, coco, model, weights, jpeg_compression=70, desired_min_size=800, threshold=0.05):
    results = []

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        # resize to desired height
        image = Image.open(image_path)

        # width, height = image.size
        # min_size = min(width, height)
        # scale_factor = desired_min_size / float(min_size)
        # scale_factor = scale_factor if scale_factor > 1 else 1
        # scaled_width = int(round(width * scale_factor))
        # scaled_height = int(round(height * scale_factor))
        # image = image.resize((scaled_width,scaled_height), Image.Resampling.LANCZOS)

        if jpeg_compression != 100:
            buffer = BytesIO()
            image.save(buffer, "JPEG", quality=jpeg_compression)
            buffer.seek(0)
            image = Image.open(buffer)

        # NB count inference time from here <---------------
        preprocess = weights.transforms()

        x = [preprocess(image).cuda()]
        #x[0].cuda()
        #print(x.size())
        #features, regression, classification, anchors = model(x)
        pred=model(x)

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
    path_to_annotations="../../../../coco2017/"
    coco_annotation_file_path = path_to_annotations+"annotations/instances_val2017.json"
    VAL_GT = path_to_annotations+"annotations/instances_val2017.json"
    VAL_IMGS = path_to_annotations+"val2017/"
    MAX_IMAGES = 50000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    SET_NAME="resnet50"
    jpeg_compression_rate = 100
    min_size = 800
    max_size = 1333
    if override_prev_results or not os.path.exists(f"{SET_NAME}_bbox_results.json"):
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9, min_size=min_size, max_size=max_size)
        model.eval()

        if use_cuda:
            model.cuda()
            print("MODEL ON CUDA")
        evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model, weights, jpeg_compression_rate)

    _eval(coco_gt, image_ids, "{}_bbox_results.json".format(SET_NAME))
    print("min_size: {}, max_size: {}, jpeg_compression_rate: {}".format(min_size, max_size, jpeg_compression_rate))

# DONE (t=0.97s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.500
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.416
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.183
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.403
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.533
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.408
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.196
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.444
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.596
# Current AP: 0.36905
# min_size: 800, max_size: 1333, jpeg_compression_rate: 100

#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.339
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.462
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.381
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.146
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.369
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.521
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.281
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.375
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.154
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.405
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.577
# Current AP: 0.33900
# min_size: 800, max_size: 1333, jpeg_compression_rate: 70