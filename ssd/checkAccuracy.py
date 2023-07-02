import argparse
from SSD_creator import SSD_creator
from evaluation_utils import *
from inference_utils import *

def get_parameters():
    parser = argparse.ArgumentParser(description='Parser for SSD300')
    parser.add_argument('-p', '--precision', default="fp32", type=str)
    parser.add_argument('-d', '--data', default="../../../coco2017", type=str)
    parser.add_argument('-a', '--amp_autocast', default=False, type=bool)
    parser.add_argument('-e', '--eval_batch_size', default=32, type=int)
    parser.add_argument('-n', '--num_workers', default=8, type=int)
    parser.add_argument('-w', '--pretrained_weigths', default="IMAGENET1K_V2", type=str)
    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    args = get_parameters()
    print(args)
    ssd_creator = SSD_creator(args['precision'], args['pretrained_weigths'])

    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)    

    val_dataset = get_val_dataset(args)
    val_dataloader = get_val_dataloader(val_dataset, args)
    inv_map = {v: k for k, v in val_dataset.label_map.items()}
    
    acc = evaluate(ssd_creator.ssd300, val_dataloader, cocoGt, encoder, inv_map, args)
    print('Model precision {} mAP'.format(acc))

# {'precision': 'fp32', 'data': './coco', 'eval_batch_size': 32, 'num_workers': 8}
# Weights already exist for fp32 precision
# network converted to fp32
# Loading COCO ground truth... {'precision': 'fp32', 'data': './coco', 'eval_batch_size': 32, 'num_workers': 8}
# loading annotations into memory...
# Done (t=0.35s)
# creating index...
# index created!
# Parsing batch: 154/155
# Predicting Ended, total time: 117.41 s
# Loading and preparing results...
# Converting ndarray to lists...
# (310908, 7)
# 0/310908
# DONE (t=1.35s)
# creating index...
# index created!
# Running per image evaluation...
# Evaluate annotation type *bbox*
# DONE (t=20.10s).
# Accumulating evaluation results...
# DONE (t=2.96s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.253
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.428
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.262
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.273
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.409
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.239
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.345
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.114
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.399
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.557
# Current AP: 0.25344
# Model precision 0.253435239881862 mAP

# {'precision': 'fp16', 'data': './coco', 'eval_batch_size': 32, 'num_workers': 8}
# Weights already exist for fp16 precision
# network converted to fp16
# Loading COCO ground truth... {'precision': 'fp16', 'data': './coco', 'eval_batch_size': 32, 'num_workers': 8}
# loading annotations into memory...
# Done (t=0.35s)
# creating index...
# index created!
# Parsing batch: 154/155
# Predicting Ended, total time: 111.83 s
# Loading and preparing results...
# Converting ndarray to lists...
# (310951, 7)
# 0/310951
# DONE (t=1.27s)
# creating index...
# index created!
# Running per image evaluation...
# Evaluate annotation type *bbox*
# DONE (t=19.01s).
# Accumulating evaluation results...
# DONE (t=2.85s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.254
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.428
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.262
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.274
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.409
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.239
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.345
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.114
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.399
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.557
# Current AP: 0.25361
# Model precision 0.25360768741276757 mAP
