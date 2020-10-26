from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
from .coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, get_label_map 
from .openimages import OpenImagesDetection, OpenImagesAnnotationTransform, OpenImages_ROOT, OPENIMAGES_CATS

from .config import *
import torch
import cv2
import numpy as np


class Collate:
    def __init__(self, word_emb=False, return_id=False):
        self.word_emb = word_emb
        if return_id:
            self.return_index = 4
        else:
            self.return_index = 3

    def detection_collate(self, batch):
        """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).

        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations

        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on
                                    0 dim
        """
        targets = []
        imgs = []
        word_emb_imgs = []
        ids = []
        for sample in batch:
            imgs.append(sample[0])
            targets.append(torch.FloatTensor(sample[1]))
            word_emb_imgs.append(sample[2])
            ids.append(sample[3])
        if self.word_emb:
            ret = torch.stack(imgs, 0), targets, torch.stack(word_emb_imgs), ids
            return ret[:self.return_index] 
        else: 
            ret = torch.stack(imgs, 0), targets, None, ids
            return ret[:self.return_index]

def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
