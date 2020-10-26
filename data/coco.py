from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from data import coco_text
import csv

COCO_ROOT = '/media/kouki/kouki/COCOtext'
# COCO_ROOT = osp.join('./', 'data/')
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
		'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
		'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
		'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
		'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
		'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
		'kite', 'baseball bat', 'baseball glove', 'skateboard', 
		'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
		'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
		'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
		'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
		'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
		'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
		'refrigerator', 'book', 'clock', 'vase', 'scissors', 
		'teddy bear', 'hair drier', 'toothbrush')


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map

def transform_bboxes(bboxes, height, width):
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return bboxes

class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self, coco, label_map):
        # self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))
        self.coco = coco
        self.label_map = label_map

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                cat = self.coco.cats[obj['category_id']]['name']
                label_idx = self.label_map[cat]
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_set='train2014', transform=None, 
                 dataset_name='MS COCO', wv_model=None, reject_words=[], 
                 size=300, cats=[]):
        sys.path.append(osp.join(root, COCO_API))
        from pycocotools.coco import COCO
        self.root = osp.join(root, IMAGES, image_set)
        self.coco = COCO(osp.join(root, ANNOTATIONS,
                                  INSTANCES_SET.format(image_set)))
        self.image_set = image_set
        self.cat_ids = self.coco.getCatIds(cats)
        self.cats = [self.coco.cats[id_]['name'] for id_ in self.cat_ids]
        if len(cats) != len(self.cats) and len(cats) != 0:
            print('class name error: invalid name in specified category.')
            raise ValueError
        self.label_map = dict([(cat, i) for i, cat in enumerate(self.cats, 1)])
        self.target_transform = COCOAnnotationTransform(self.coco, self.label_map)
        os.makedirs('label_file/coco', exist_ok=True)
        with open('label_file/coco/label_file.txt', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['objectID', 'catID', 'catName'])
            for cat, cat_id in zip(self.cats, self.cat_ids):
                label_id = self.label_map[cat]
                writer.writerow([label_id, cat_id, cat])
                
        self.ids = self.get_ids()
        self.transform = transform
        self.name = dataset_name
        self.wv_model = wv_model
        self.reject_words = reject_words
        self.size = size
        if self.wv_model:
            self.ct = coco_text.COCO_Text('/media/kouki/kouki/COCOtext/text_annotations/cocotext.v2.json')
            self.preprocess()
            self.ids = list(set(self.ct.imgToAnns.keys()) & set(self.ids))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, _, _, word_emb_img = self.pull_item(index)
        return im, gt, word_emb_img, self.ids[index]

    def __len__(self):
        return len(self.ids)

    def preprocess(self):
        for key in list(self.ct.anns.keys()):
            self.ct.anns[key]['utf8_string'] = self.ct.anns[key]['utf8_string'].lower()
            if self.ct.anns[key]['utf8_string'] not in self.wv_model.vocab \
                    or self.ct.anns[key]['utf8_string'] in self.reject_words:
                del self.ct.anns[key]
        
        annIds = set(self.ct.anns.keys())
        for key in list(self.ct.imgToAnns.keys()):
            self.ct.imgToAnns[key] = list(set(self.ct.imgToAnns[key]) & annIds)
            if not self.ct.imgToAnns[key]:
                if self.ct.imgs[key]["set"] == "train":
                    self.ct.train.remove(key)
                elif self.ct.imgs[key]["set"] == "val":
                    self.ct.val.remove(key)
                else:
                    print("This id is not included in both train and val.")
                    raise ValueError
                del self.ct.imgToAnns[key]
                del self.ct.imgs[key]
    
    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        # target = self.coco.imgToAnns[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)

        target = self.coco.loadAnns(ann_ids)
        path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(path)
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.wv_model is not None: # word_emb_img debug ok
            word_emb_img = self.pull_spatial_word_emb(img_id, height, width)
        else:
            word_emb_img = None
        
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels, word_emb_img = self.transform(img, target[:, :4],
                                                target[:, 4], word_emb_img)
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        if self.wv_model is not None:
            word_emb_img = torch.from_numpy(word_emb_img).permute(2, 0, 1)
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, target, height, width, word_emb_img
                

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)
        
    def pull_spatial_word_emb(self, img_id, height, width):
        annIds = self.ct.getAnnIds(imgIds=img_id)
        anns = self.ct.loadAnns(annIds)
        bboxes = transform_bboxes(np.array([ann['bbox'] for ann in anns]), height, width)
        words = [ann['utf8_string'] for ann in anns]
        word_vectors = [self.wv_model[word] for word in words]
        word_emb_img = np.zeros((height, width, self.wv_model.vector_size))
        for bbox, vector in zip(bboxes, word_vectors):
            xmin, ymin, xmax, ymax = bbox.astype(np.int)
            word_emb_img[ymin:ymax, xmin:xmax, :] += vector
        return word_emb_img

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def get_ids(self):
        s = set()
        for cat in self.cat_ids:
            s = s | set(self.coco.getImgIds(catIds=cat)) 
        return list(s)