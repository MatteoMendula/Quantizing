import os
import json
import time
import bz2
import pickle
import random
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
from pycocotools.coco import COCO

from inference_utils import *

# Implement a datareader for COCO dataset
class COCODetection(data.Dataset):
    def __init__(self, img_folder, annotate_file, transform=None):
        self.img_folder = img_folder
        self.annotate_file = annotate_file

        # Start processing annotation
        with open(annotate_file) as fin:
            self.data = json.load(fin)

        self.images = {}

        self.label_map = {}
        self.label_info = {}
        start_time = time.time()
        # 0 stand for the background
        cnt = 0
        self.label_info[cnt] = "background"
        for cat in self.data["categories"]:
            cnt += 1
            self.label_map[cat["id"]] = cnt
            self.label_info[cnt] = cat["name"]

        # build inference for images
        for img in self.data["images"]:
            img_id = img["id"]
            img_name = img["file_name"]
            img_size = (img["height"],img["width"])
            if img_id in self.images: raise Exception("dulpicated image record")
            self.images[img_id] = (img_name, img_size, [])

        # read bboxes
        for bboxes in self.data["annotations"]:
            img_id = bboxes["image_id"]
            category_id = bboxes["category_id"]
            bbox = bboxes["bbox"]
            bbox_label = self.label_map[bboxes["category_id"]]
            self.images[img_id][2].append((bbox, bbox_label))

        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                self.images.pop(k)

        self.img_keys = list(self.images.keys())
        self.transform = transform

    @property
    def labelnum(self):
        return len(self.label_info)

    @staticmethod
    def load(pklfile):
        with bz2.open(pklfile, "rb") as fin:
            ret = pickle.load(fin)
        return ret

    def save(self, pklfile):
        with bz2.open(pklfile, "wb") as fout:
            pickle.dump(self, fout)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.img_keys[idx]
        img_data = self.images[img_id]
        fn = img_data[0]
        img_path = os.path.join(self.img_folder, fn)
        img = Image.open(img_path).convert("RGB")

        htot, wtot = img_data[1]
        bbox_sizes = []
        bbox_labels = []

        #for (xc, yc, w, h), bbox_label in img_data[2]:
        for (l,t,w,h), bbox_label in img_data[2]:
            r = l + w
            b = t + h
            #l, t, r, b = xc - 0.5*w, yc - 0.5*h, xc + 0.5*w, yc + 0.5*h
            bbox_size = (l/wtot, t/htot, r/wtot, b/htot)
            bbox_sizes.append(bbox_size)
            bbox_labels.append(bbox_label)

        bbox_sizes = torch.tensor(bbox_sizes)
        bbox_labels =  torch.tensor(bbox_labels)


        if self.transform != None:
            img, (htot, wtot), bbox_sizes, bbox_labels = \
                self.transform(img, (htot, wtot), bbox_sizes, bbox_labels)
        else:
            pass

        return img, img_id, (htot, wtot), bbox_sizes, bbox_labels



# This function is from https://github.com/chauhan-utk/ssd.DomainAdaptation.
class SSDCropping(object):
    """ Cropping for SSD, according to original paper
        Choose between following 3 conditions:
        1. Preserve the original image
        2. Random crop minimum IoU is among 0.1, 0.3, 0.5, 0.7, 0.9
        3. Random crop
        Reference to https://github.com/chauhan-utk/src.DomainAdaptation
    """
    def __init__(self):

        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )

    def __call__(self, img, img_size, bboxes, labels):

        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)

            if mode is None:
                return img, img_size, bboxes, labels

            htot, wtot = img_size

            min_iou, max_iou = mode
            min_iou = float("-inf") if min_iou is None else min_iou
            max_iou = float("+inf") if max_iou is None else max_iou

            # Implementation use 50 iteration to find possible candidate
            for _ in range(1):
                # suze of each sampled path in [0.1, 1] 0.3*0.3 approx. 0.1
                w = random.uniform(0.3 , 1.0)
                h = random.uniform(0.3 , 1.0)

                if w/h < 0.5 or w/h > 2:
                    continue

                # left 0 ~ wtot - w, top 0 ~ htot - h
                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                ious = calc_iou_tensor(bboxes, torch.tensor([[left, top, right, bottom]]))

                # tailor all the bboxes and return
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bboxes whose center not in the cropped image
                xc = 0.5*(bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5*(bboxes[:, 1] + bboxes[:, 3])

                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # if no such boxes, continue searching again
                if not masks.any():
                    continue

                bboxes[bboxes[:, 0] < left, 0] = left
                bboxes[bboxes[:, 1] < top, 1] = top
                bboxes[bboxes[:, 2] > right, 2] = right
                bboxes[bboxes[:, 3] > bottom, 3] = bottom

                bboxes = bboxes[masks, :]
                labels = labels[masks]

                left_idx = int(left*wtot)
                top_idx =  int(top*htot)
                right_idx = int(right*wtot)
                bottom_idx = int(bottom*htot)
                img = img.crop((left_idx, top_idx, right_idx, bottom_idx))

                bboxes[:, 0] = (bboxes[:, 0] - left)/w
                bboxes[:, 1] = (bboxes[:, 1] - top)/h
                bboxes[:, 2] = (bboxes[:, 2] - left)/w
                bboxes[:, 3] = (bboxes[:, 3] - top)/h

                htot = bottom_idx - top_idx
                wtot = right_idx - left_idx
                return img, (htot, wtot), bboxes, labels

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, bboxes):
        if random.random() < self.p:
            bboxes[:, 0], bboxes[:, 2] = 1.0 - bboxes[:, 2], 1.0 - bboxes[:, 0]
            return image.transpose(Image.FLIP_LEFT_RIGHT), bboxes
        return image, bboxes

# Do data augumentation
class SSDTransformer(object):
    """ SSD Data Augumentation, according to original paper
        Composed by several steps:
        Cropping
        Resize
        Flipping
        Jittering
    """
    def __init__(self, dboxes, size = (300, 300), val=False):

        # define vgg16 mean
        self.size = size
        self.val = val

        self.dboxes_ = dboxes #DefaultBoxes300()
        self.encoder = Encoder(self.dboxes_)

        self.crop = SSDCropping()
        self.img_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ColorJitter(brightness=0.125, contrast=0.5,
                saturation=0.5, hue=0.05
            ),
            transforms.ToTensor()
        ])
        self.hflip = RandomHorizontalFlip()

        # All Pytorch Tensor will be normalized
        # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        self.trans_val = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            #ToTensor(),
            self.normalize,])

    @property
    def dboxes(self):
        return self.dboxes_

    def __call__(self, img, img_size, bbox=None, label=None, max_num=200):
        #img = torch.tensor(img)
        if self.val:
            bbox_out = torch.zeros(max_num, 4)
            label_out =  torch.zeros(max_num, dtype=torch.long)
            bbox_out[:bbox.size(0), :] = bbox
            label_out[:label.size(0)] = label
            return self.trans_val(img), img_size, bbox_out, label_out

        img, img_size, bbox, label = self.crop(img, img_size, bbox, label)
        img, bbox = self.hflip(img, bbox)

        img = self.img_trans(img).contiguous()
        img = self.normalize(img)

        bbox, label = self.encoder.encode(bbox, label)

        return img, img_size, bbox, label

def get_val_dataset(args):
    dboxes = dboxes300_coco()
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)

    val_annotate = os.path.join(args["data"], "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args["data"], "val2017")

    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    return val_coco

def get_val_dataloader(dataset, args):
    val_sampler = None
    val_dataloader = data.DataLoader(dataset,
                                batch_size=args["eval_batch_size"],
                                shuffle=False,  # Note: distributed sampler is shuffled :(
                                sampler=val_sampler,
                                num_workers=args["num_workers"])

    return val_dataloader

def get_coco_ground_truth(args):
    print("Loading COCO ground truth...", args)
    val_annotate = os.path.join(args["data"], "annotations/instances_val2017.json")
    cocoGt = COCO(annotation_file=val_annotate)
    return cocoGt

