# from .config import HOME
# import os
# import os.path as osp
# import sys
# import torch
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import cv2
# import numpy as np
# from data import coco_text
# import csv
# import pandas as pd
# from glob import glob
# import json



# OpenImages_ROOT = '/media/kouki/kouki/Openimages_dataset'

# class OpenImagesAnnotationTransform(object):
#     """Transforms a COCO annotation into a Tensor of bbox coords and label index
#     Initilized with a dictionary lookup of classnames to indexes
#     """
#     def __init__(self, label_map):
#         # self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))
#         self.label_map = label_map

#     def __call__(self, obj_anns):
#         """
#         Args:
#             target (dict): COCO target json annotation as a python dict
#             height (int): height
#             width (int): width
#         Returns:
#             a list containing lists of bounding boxes  [bbox coords, class idx]
#         """
#         cords = obj_anns.loc[:, 'XMin':'YMax'].values
#         cat_ids = obj_anns['CatName'].map(self.label_map).values
#         res = np.concatenate([cords, cat_ids[:, np.newaxis]], axis=1)
#         return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]

# OPENIMAGES_ROOT = '/media/kouki/kouki/Openimages_dataset'
# IMAGES = 'images'
# ANNOTATIONS = 'annotations'

# class OpenImagesDetection(data.Dataset):
#     """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
#     Args:
#         root (string): Root directory where images are downloaded to.
#         set_name (string): Name of the specific set of COCO images.
#         transform (callable, optional): A function/transform that augments the
#                                         raw images`
#         target_transform (callable, optional): A function/transform that takes
#         in the target (bbox) and transforms it.
#     """

#     def __init__(self, root, image_set='train', transform=None, 
#                  dataset_name='OpenImages', wv_model=None, reject_words=[], 
#                  size=300, cats=[], text_root='/media/kouki/kouki/scene_text_data/'):
#         self.root = osp.join(root, image_set)
#         self.image_set = image_set
#         self.annotations = self.get_object_anns(root, cats)
#         self.label_map = self.get_labelmap(root)
#         self.text_root = text_root

#         self.cats = [cat for label_id, cat in sorted(self.label_map.items())]
#         if len(cats) != self.cats and len(cats) != 0:
#             print('class name error: invalid name in specified category.')
#             raise ValueError

#         self.ids = self.get_ids()
#         self.transform = transform
#         self.target_transform = OpenImagesAnnotationTransform(self.label_map)
#         self.name = dataset_name
#         self.wv_model = wv_model
#         self.reject_words = reject_words
#         self.size = size
#         if self.wv_model:
#             ann_dir = os.path.join(self.text_root, 'bbox_data', 'OpenImages', image_set)
#             self.text_ann = get_pkl_text_ann(ann_dir, self.reject_words)
#             self.ids = list(set(self.text_ann['ImageID']) & set(self.ids))

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: Tuple (image, target).
#                    target is the object returned by ``coco.loadAnns``.
#         """
#         im, gt, _, _, word_emb_img = self.pull_item(index)
#         return im, gt, word_emb_img

#     def __len__(self):
#         return len(self.ids)

    
#     def pull_item(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: Tuple (image, target, height, width).
#                    target is the object returned by ``coco.loadAnns``.
#         """
#         img_id = self.ids[index]

#         obj_anns = self.annotations[self.annotations['ImageID'] == img_id]
#         text_anns = self.text_ann[self.text_ann['ImageID'] == img_id]
        
#         path = osp.join(self.root, '*', img_id + '.jpg')
#         assert osp.exists(path), 'Image path does not exist: {}'.format(path)
#         img = cv2.imread(path)
#         height, width, _ = img.shape
#         if self.target_transform is not None:
#             target = self.target_transform(obj_anns)
            
#         if self.wv_model is not None: # word_emb_img debug ok
#             word_emb_img = self.pull_spatial_word_emb(text_anns, height, width)
#         else:
#             word_emb_img = None
        
#         if self.transform is not None:
#             # target = np.array(target)
#             img, boxes, labels, word_emb_img = self.transform(img, target[:, :4],
#                                                 target[:, 4], word_emb_img)
#             # to rgb
#             img = img[:, :, (2, 1, 0)]
#             target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

#         if self.wv_model is not None:
#             word_emb_img = torch.from_numpy(word_emb_img).permute(2, 0, 1)
#         img = torch.from_numpy(img).permute(2, 0, 1)

#         return img, target, height, width, word_emb_img
                

#     def pull_image(self, index):
#         '''Returns the original image object at index in PIL form

#         Note: not using self.__getitem__(), as any transformations passed in
#         could mess up this functionality.

#         Argument:
#             index (int): index of img to show
#         Return:
#             cv2 img
#         '''
#         # img_id = self.ids[index]
#         # path = self.coco.loadImgs(img_id)[0]['file_name']
#         # return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)
#         pass

#     def pull_spatial_word_emb(self, text_ann, height, width):
#         bboxes = text_ann.loc[:, 'x0':'y3'].values.reshape((-1, 4, 2))
#         words = text_ann['Text']
#         word_vectors = [self.wv_model[word] for word in words]
#         word_emb_img = np.zeros((height, width, self.wv_model.vector_size))
#         for bbox, vector in zip(bboxes, word_vectors):
#             # 要デバッグ（BBoxのx座標とy座標がどうなってるか確認する．cv2.fillPolyのptsは[[x0（横）, y0（縦）], ...]の順序．）
#             mask = cv2.fillPoly(np.zeros(height, width), [bbox], 1)
#             word_emb_img +=  vector * mask[:, :, np.newaxis]
#         return word_emb_img

#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         fmt_str += '    Root Location: {}\n'.format(self.root)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         tmp = '    Target Transforms (if any): '
#         fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str

#     def get_ids(self):
#         image_ids = set(self.annotations['ImageID'])
#         return list(image_ids)

#     def get_labelmap(self, root):
#         cats = set(self.annotations['CatName'])
#         label_file = os.path.join('label_file', 'openimages', 'label_file.txt')
#         os.makedirs(label_file, exist_ok=True)
#         with open(label_file, 'w') as f:
#             writer = csv.writer(f)
#             writer.writerows([(i, cat) for i, cat in enumerate(cats, 1)])
#         return dict([(cat, i) for i, cat in enumerate(cats, 1)])

#     def get_object_anns(self, root, cats):
#         label_discription_path = os.path.join(root, 'class-descriptions-boxable.csv')
#         label_disc = pd.read_csv(label_discription_path, names=['LabelName', 'CatName'])
#         if cats:
#             label_disc = label_disc[label_disc['CatName'].isin(cats)]

#         ann_dir = os.path.join(root, ANNOTATIONS)
#         ann_file = os.path.join(ann_dir, self.image_set + '-annotations-bbox.csv')
#         use_cols=['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']
#         ann = pd.read_csv(ann_file, usecols=use_cols)

#         ann = pd.merge(label_disc, ann, on='LabelName').drop('LabelName', axis=1)
#         return ann
    

# def get_pkl_text_ann(ann_dir, reject_words):
#     ann_paths = glob(os.path.join(ann_dir, '*.pkl'))
#     anns = [pd.read_pickle(path) for path in ann_paths]
#     ann = pd.concat(anns)
#     ann = ann[~ann['Text'].isin(reject_words)]
#     return ann

# def get_json_text_ann(ann_dir):
#     ann_paths = glob(os.path.join(ann_dir, '*.json'))
#     ann = {}
#     for ann_path in ann_paths:
#         with open(ann_path) as f:
#             ann_sub = json.load(f)
#             ann.update(ann_sub)
#     return ann



        
