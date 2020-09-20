# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
import torch.utils.data as data
from ssd import build_ssd

from data import *
import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt




if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_COCO_231.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=COCO_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

if args.voc_root == COCO_ROOT:
    from data import COCO_CLASSES as labelmap
elif args.voc_root == VOC_ROOT:
    from data import VOC_CLASSES as labelmap

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls_label in enumerate(labelmap):
        print('Writing {:s} COCO results file'.format(cls_label))
        filename = get_voc_results_file_template(set_type, cls_label)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap




def object_detect(net, cuda, image, transform, im_size=300, thresh=0.05):
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[]for _ in range(len(labelmap)+1)]
    h, w, _ = image.shape

    # imageのサイズを300に合わせる
    x, _, _ = transform(image)
    x = x[:, :, (2, 1, 0)]
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if cuda:
        x = x.cuda()
    detections = net(x)

    scores, bboxes = detections[0].data, detections[1].data
    ##### 画像サイズを0~1からピクセル値に変換
    h, w = image.shape[:2]
    bboxes[:, 0] = (bboxes[:, 0] * w).int()
    bboxes[:, 2] = (bboxes[:, 2] * w).int()
    bboxes[:, 1] = (bboxes[:, 1] * h).int()
    bboxes[:, 3] = (bboxes[:, 3] * h).int()

    # bboxes[:] = [x1, y1, x2, y2]
    # scores[:] = [81(softmax)]
    return bboxes, scores

def object_detect_all_boxes(net, cuda, image, transform, im_size=300, thresh=0.05):
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[]for _ in range(len(labelmap)+1)]
    h, w, _ = image.shape

    # imageのサイズを300に合わせる
    x, _, _ = transform(image)
    x = x[:, :, (2, 1, 0)]
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if args.cuda:
        x = x.cuda()

    detections = net(x).data
    

    # skip j = 0, because it's the background class
    for j in range(1, detections.size(1)):
        dets = detections[0, j, :]
        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)
        if dets.size(0) == 0:
            continue
        boxes = dets[:, 1:]
        boxes[:, 0] *= w
        boxes[:, 2] *= w
        boxes[:, 1] *= h
        boxes[:, 3] *= h
        scores = dets[:, 0].cpu().numpy()
        cls_dets = np.hstack((boxes.cpu().numpy(),
                                scores[:, np.newaxis])).astype(np.float32,
                                                                copy=False)
        all_boxes[j] = cls_dets

    return all_boxes

def threshold_bbox(all_boxes, threshold = 0.5):
    pts_list = []
    for j in range(1,len(labelmap) + 1):
        cls_dets = all_boxes[j]
        scores = cls_dets[:,-1]
        for det_idx, score in enumerate(scores):
            if score > 0.3:
                Xmin, Xmax = np.int32(cls_dets[det_idx][0]), np.int32(cls_dets[det_idx][2])
                Ymin, Ymax = np.int32(cls_dets[det_idx][1]), np.int32(cls_dets[det_idx][3])
                # pts = np.array(((Xmin, Ymin),(Xmin, Ymax),(Xmax, Ymax),(Xmax, Ymin), j))
                pts = np.array((Xmin, Ymin, Xmax, Ymax, j))
                pts_list.append(pts)

    return np.array(pts_list), np.array(scores)


# bbos描写用
# image = cv2.imread(os.path.join(dataset.root, path), cv2.IMREAD_COLOR)
# cv2.polylines(image, pts_list, True, (255, 255, 255), thickness=2)
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()










################################## BACK Up ###################################
# # -*- coding: utf-8 -*-
# from __future__ import print_function
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# from torch.autograd import Variable
# from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
# import torch.utils.data as data
# from ssd import build_ssd

# from data import *
# import sys
# import os
# import time
# import argparse
# import numpy as np
# import pickle
# import cv2
# import matplotlib.pyplot as plt


# # added
# import torch.nn.functional as F
# # Word2vec
# from gensim.models import word2vec, Word2Vec
# from gensim.models import KeyedVectors

# if sys.version_info[0] == 2:
#     import xml.etree.cElementTree as ET
# else:
#     import xml.etree.ElementTree as ET

# def str2bool(v):
#     return v.lower() in ("yes", "true", "t", "1")


# parser = argparse.ArgumentParser(
#     description='Single Shot MultiBox Detector Evaluation')
# parser.add_argument('--trained_model',
#                     default='weights/ssd300_COCO_231.pth', type=str,
#                     help='Trained state_dict file path to open')
# parser.add_argument('--save_folder', default='eval/', type=str,
#                     help='File path to save results')
# parser.add_argument('--confidence_threshold', default=0.01, type=float,
#                     help='Detection confidence threshold')
# parser.add_argument('--top_k', default=5, type=int,
#                     help='Further restrict the number of predictions to parse')
# parser.add_argument('--cuda', default=False, type=str2bool,
#                     help='Use cuda to train model')
# parser.add_argument('--voc_root', default=COCO_ROOT,
#                     help='Location of VOC root directory')
# parser.add_argument('--cleanup', default=True, type=str2bool,
#                     help='Cleanup and remove results files following eval')

# args = parser.parse_args()

# if args.voc_root == COCO_ROOT:
#     from data import COCO_CLASSES as labelmap
# elif args.voc_root == VOC_ROOT:
#     from data import VOC_CLASSES as labelmap

# if not os.path.exists(args.save_folder):
#     os.mkdir(args.save_folder)

# if torch.cuda.is_available():
#     if args.cuda:
#         torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     if not args.cuda:
#         print("WARNING: It looks like you have a CUDA device, but aren't using \
#               CUDA.  Run with --cuda for optimal eval speed.")
#         torch.set_default_tensor_type('torch.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')

# annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
# imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
# imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
#                           'Main', '{:s}.txt')
# YEAR = '2007'
# devkit_path = args.voc_root + 'VOC' + YEAR
# dataset_mean = (104, 117, 123)
# set_type = 'test'


# class Timer(object):
#     """A simple timer."""
#     def __init__(self):
#         self.total_time = 0.
#         self.calls = 0
#         self.start_time = 0.
#         self.diff = 0.
#         self.average_time = 0.

#     def tic(self):
#         # using time.time instead of time.clock because time time.clock
#         # does not normalize for multithreading
#         self.start_time = time.time()

#     def toc(self, average=True):
#         self.diff = time.time() - self.start_time
#         self.total_time += self.diff
#         self.calls += 1
#         self.average_time = self.total_time / self.calls
#         if average:
#             return self.average_time
#         else:
#             return self.diff


# def parse_rec(filename):
#     """ Parse a PASCAL VOC xml file """
#     tree = ET.parse(filename)
#     objects = []
#     for obj in tree.findall('object'):
#         obj_struct = {}
#         obj_struct['name'] = obj.find('name').text
#         obj_struct['pose'] = obj.find('pose').text
#         obj_struct['truncated'] = int(obj.find('truncated').text)
#         obj_struct['difficult'] = int(obj.find('difficult').text)
#         bbox = obj.find('bndbox')
#         obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
#                               int(bbox.find('ymin').text) - 1,
#                               int(bbox.find('xmax').text) - 1,
#                               int(bbox.find('ymax').text) - 1]
#         objects.append(obj_struct)

#     return objects


# def get_output_dir(name, phase):
#     """Return the directory where experimental artifacts are placed.
#     If the directory does not exist, it is created.
#     A canonical path is built using the name from an imdb and a network
#     (if not None).
#     """
#     filedir = os.path.join(name, phase)
#     if not os.path.exists(filedir):
#         os.makedirs(filedir)
#     return filedir


# def get_voc_results_file_template(image_set, cls):
#     # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
#     filename = 'det_' + image_set + '_%s.txt' % (cls)
#     filedir = os.path.join(devkit_path, 'results')
#     if not os.path.exists(filedir):
#         os.makedirs(filedir)
#     path = os.path.join(filedir, filename)
#     return path


# def write_voc_results_file(all_boxes, dataset):
#     for cls_ind, cls_label in enumerate(labelmap):
#         print('Writing {:s} COCO results file'.format(cls_label))
#         filename = get_voc_results_file_template(set_type, cls_label)
#         with open(filename, 'wt') as f:
#             for im_ind, index in enumerate(dataset.ids):
#                 dets = all_boxes[cls_ind+1][im_ind]
#                 if dets == []:
#                     continue
#                 # the VOCdevkit expects 1-based indices
#                 for k in range(dets.shape[0]):
#                     f.write('{:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
#                             format(index, dets[k, -1],
#                                    dets[k, 0] + 1, dets[k, 1] + 1,
#                                    dets[k, 2] + 1, dets[k, 3] + 1))


# def do_python_eval(output_dir='output', use_07=True):
#     cachedir = os.path.join(devkit_path, 'annotations_cache')
#     aps = []
#     # The PASCAL VOC metric changed in 2010
#     use_07_metric = use_07
#     print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
#     if not os.path.isdir(output_dir):
#         os.mkdir(output_dir)
#     for i, cls in enumerate(labelmap):
#         filename = get_voc_results_file_template(set_type, cls)
#         rec, prec, ap = voc_eval(
#            filename, annopath, imgsetpath.format(set_type), cls, cachedir,
#            ovthresh=0.5, use_07_metric=use_07_metric)
#         aps += [ap]
#         print('AP for {} = {:.4f}'.format(cls, ap))
#         with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
#             pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
#     print('Mean AP = {:.4f}'.format(np.mean(aps)))
#     print('~~~~~~~~')
#     print('Results:')
#     for ap in aps:
#         print('{:.3f}'.format(ap))
#     print('{:.3f}'.format(np.mean(aps)))
#     print('~~~~~~~~')
#     print('')
#     print('--------------------------------------------------------------')
#     print('Results computed with the **unofficial** Python eval code.')
#     print('Results should be very close to the official MATLAB eval code.')
#     print('--------------------------------------------------------------')


# def voc_ap(rec, prec, use_07_metric=True):
#     """ ap = voc_ap(rec, prec, [use_07_metric])
#     Compute VOC AP given precision and recall.
#     If use_07_metric is true, uses the
#     VOC 07 11 point method (default:True).
#     """
#     if use_07_metric:
#         # 11 point metric
#         ap = 0.
#         for t in np.arange(0., 1.1, 0.1):
#             if np.sum(rec >= t) == 0:
#                 p = 0
#             else:
#                 p = np.max(prec[rec >= t])
#             ap = ap + p / 11.
#     else:
#         # correct AP calculation
#         # first append sentinel values at the end
#         mrec = np.concatenate(([0.], rec, [1.]))
#         mpre = np.concatenate(([0.], prec, [0.]))

#         # compute the precision envelope
#         for i in range(mpre.size - 1, 0, -1):
#             mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

#         # to calculate area under PR curve, look for points
#         # where X axis (recall) changes value
#         i = np.where(mrec[1:] != mrec[:-1])[0]

#         # and sum (\Delta recall) * prec
#         ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#     return ap


# def voc_eval(detpath,
#              annopath,
#              imagesetfile,
#              classname,
#              cachedir,
#              ovthresh=0.5,
#              use_07_metric=True):
#     """rec, prec, ap = voc_eval(detpath,
#                            annopath,
#                            imagesetfile,
#                            classname,
#                            [ovthresh],
#                            [use_07_metric])
# Top level function that does the PASCAL VOC evaluation.
# detpath: Path to detections
#    detpath.format(classname) should produce the detection results file.
# annopath: Path to annotations
#    annopath.format(imagename) should be the xml annotations file.
# imagesetfile: Text file containing the list of images, one image per line.
# classname: Category name (duh)
# cachedir: Directory for caching the annotations
# [ovthresh]: Overlap threshold (default = 0.5)
# [use_07_metric]: Whether to use VOC07's 11 point AP computation
#    (default True)
# """
# # assumes detections are in detpath.format(classname)
# # assumes annotations are in annopath.format(imagename)
# # assumes imagesetfile is a text file with each line an image name
# # cachedir caches the annotations in a pickle file
# # first load gt
#     if not os.path.isdir(cachedir):
#         os.mkdir(cachedir)
#     cachefile = os.path.join(cachedir, 'annots.pkl')
#     # read list of images
#     with open(imagesetfile, 'r') as f:
#         lines = f.readlines()
#     imagenames = [x.strip() for x in lines]
#     if not os.path.isfile(cachefile):
#         # load annots
#         recs = {}
#         for i, imagename in enumerate(imagenames):
#             recs[imagename] = parse_rec(annopath % (imagename))
#             if i % 100 == 0:
#                 print('Reading annotation for {:d}/{:d}'.format(
#                    i + 1, len(imagenames)))
#         # save
#         print('Saving cached annotations to {:s}'.format(cachefile))
#         with open(cachefile, 'wb') as f:
#             pickle.dump(recs, f)
#     else:
#         # load
#         with open(cachefile, 'rb') as f:
#             recs = pickle.load(f)

#     # extract gt objects for this class
#     class_recs = {}
#     npos = 0
#     for imagename in imagenames:
#         R = [obj for obj in recs[imagename] if obj['name'] == classname]
#         bbox = np.array([x['bbox'] for x in R])
#         difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
#         det = [False] * len(R)
#         npos = npos + sum(~difficult)
#         class_recs[imagename] = {'bbox': bbox,
#                                  'difficult': difficult,
#                                  'det': det}

#     # read dets
#     detfile = detpath.format(classname)
#     with open(detfile, 'r') as f:
#         lines = f.readlines()
#     if any(lines) == 1:

#         splitlines = [x.strip().split(' ') for x in lines]
#         image_ids = [x[0] for x in splitlines]
#         confidence = np.array([float(x[1]) for x in splitlines])
#         BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

#         # sort by confidence
#         sorted_ind = np.argsort(-confidence)
#         sorted_scores = np.sort(-confidence)
#         BB = BB[sorted_ind, :]
#         image_ids = [image_ids[x] for x in sorted_ind]

#         # go down dets and mark TPs and FPs
#         nd = len(image_ids)
#         tp = np.zeros(nd)
#         fp = np.zeros(nd)
#         for d in range(nd):
#             R = class_recs[image_ids[d]]
#             bb = BB[d, :].astype(float)
#             ovmax = -np.inf
#             BBGT = R['bbox'].astype(float)
#             if BBGT.size > 0:
#                 # compute overlaps
#                 # intersection
#                 ixmin = np.maximum(BBGT[:, 0], bb[0])
#                 iymin = np.maximum(BBGT[:, 1], bb[1])
#                 ixmax = np.minimum(BBGT[:, 2], bb[2])
#                 iymax = np.minimum(BBGT[:, 3], bb[3])
#                 iw = np.maximum(ixmax - ixmin, 0.)
#                 ih = np.maximum(iymax - iymin, 0.)
#                 inters = iw * ih
#                 uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
#                        (BBGT[:, 2] - BBGT[:, 0]) *
#                        (BBGT[:, 3] - BBGT[:, 1]) - inters)
#                 overlaps = inters / uni
#                 ovmax = np.max(overlaps)
#                 jmax = np.argmax(overlaps)

#             if ovmax > ovthresh:
#                 if not R['difficult'][jmax]:
#                     if not R['det'][jmax]:
#                         tp[d] = 1.
#                         R['det'][jmax] = 1
#                     else:
#                         fp[d] = 1.
#             else:
#                 fp[d] = 1.

#         # compute precision recall
#         fp = np.cumsum(fp)
#         tp = np.cumsum(tp)
#         rec = tp / float(npos)
#         # avoid divide by zero in case the first detection matches a difficult
#         # ground truth
#         prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#         ap = voc_ap(rec, prec, use_07_metric)
#     else:
#         rec = -1.
#         prec = -1.
#         ap = -1.

#     return rec, prec, ap


# def position_count_datas(model, positions, object, text, k):
#     vocab = model.wv.vocab
#     if not object in vocab:
#         print('{} is not in vocabulary.'.format(object))
#         return None
#     elif not text in vocab:
#         print('{} is not in vocabulary.'.format(text))
#         return None
#     else:
#         position = np.array(np.r_[model[object], model[text]])
#         a = []
#         l = []
#         l_a_list = []
#         i = 0
#         for data in positions:
#             a.append(data[-1])
#             l.append(np.linalg.norm(data[0:-1] - position))
#         while i < k:
#             index = np.argmin(np.array(l))
#             l_a_list.append((l[index], a[index]))
#             del l[index]
#             del a[index]
#             i += 1
#         return l_a_list


# def Probability(l_a_list, sigma2 = 1):
#     sum = 0
#     for l_a in l_a_list:
#         sum += l_a[1] * np.exp(-l_a[0]**2/2/sigma2)
#     P = sum / (2 * np.pi) ** 200
#     return P

# def inbox_mask(Classbbox_posi, Textbboxes_posi, scene_texts):  # 引数： object bbox: np.array([Xmim, Xmax, Ymin, Ymax]) text bbox: torch.tensor([[[x1,y1], ...,[x3,y3]], ...])
#     Textbbox_posi_x = Textbbox_posi[:,0]   # 戻り値： ClassbboxにTextbboxが完全に入ってたらTrue,それ以外はFalse のリスト
#     Textbbox_posi_y = Textbbox_posi[:,1]
#     mask_x_for_each_point = torch.tensor([[Classbbox_posi[0] <= k <= Classbbox_posi[1] for k in l] for l in Textbbox_posi_x])
#     mask_y_for_each_point = torch.tensor([[Classbbox_posi[2] <= k <= Classbbox_posi[3] for k in l] for l in Textbbox_posi_y])

#     mask_x = torch.prod(mask_x_for_each_point, axis=1)
#     mask_y = torch.prod(mask_y_for_each_point, axis=1)
    
#     mask_text = (mask_x * mask_y).bool()
#     mask_box = mask_text.unsqueeze(0).t().unsqueeze(1).expand(-1, 4, 2)

#     return torch.masked_select(Textbboxes_posi, mask_box).view(-1, 4, 2), torch.masked_select(scene_texts, mask_text)

# def text_score(model, detections, text_bboxes, scene_texts, position_data):
#     text_score = torch.tensor((detections.size(0), detections.size(1), detections.size(2)))
#     for cls_ in detections.size(1):
#         for object_bboxes in detections[0, cls_]:
#             for object_bbox in object_bboxes[:, 1:]:
#                 inbox_masked_text_bboxes = inbox_mask(object_bbox, text_bboxes)
#                 P_txt = Probability_txt(model, inbox_masked_text_bboxes, scene_texts, position_data)

#     return 0


# def Probability_txt(model, text_bboxes, scene_texts, position_data, num_classes = 81):
#     Pmax = torch.zeros(num_classes)
#     for scene_text in scene_texts:
#         for cls_ in range(1, num_classes):
#             pds = position_count_datas(model, position_data, scene_text, labelmap[cls_- 1])
#             if pds != None and Pmax[cls_- 1] < Probability(pds):
#                 Pmax[cls_- 1] = Probability(pds)
#     Pmax = torch.softmax(Pmax)
#     return F.softmax(Pmax.float())

                



# def object_detect(net, cuda, image, transform, im_size=300, thresh=0.05):
#     # all detections are collected into:
#     #    all_boxes[cls][image] = N x 5 array of detections in
#     #    (x1, y1, x2, y2, score)
#     all_boxes = [[]for _ in range(len(labelmap)+1)]
#     h, w, _ = image.shape

#     # imageのサイズを300に合わせる
#     x, _, _ = transform(image)
#     x = x[:, :, (2, 1, 0)]
#     x = torch.from_numpy(x).permute(2, 0, 1)
#     x = Variable(x.unsqueeze(0))
#     if args.cuda:
#         x = x.cuda()

#     detections = net(x).data



#     # skip j = 0, because it's the background class
#     pts_list = [] # bbox表示用
#     for j in range(1, detections.size(1)):
#         dets = detections[0, j, :]
#         mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
#         dets = torch.masked_select(dets, mask).view(-1, 5)
#         if dets.size(0) == 0:
#             continue
#         boxes = dets[:, 1:]
#         boxes[:, 0] *= w
#         boxes[:, 2] *= w
#         boxes[:, 1] *= h
#         boxes[:, 3] *= h
#         scores = dets[:, 0].cpu().numpy()
#         cls_dets = np.hstack((boxes.cpu().numpy(),
#                                 scores[:, np.newaxis])).astype(np.float32,
#                                                                 copy=False)
#         all_boxes[j] = cls_dets

#         ####スコアが何を示すかわからんので確認しよう
#         for det_idx, score in enumerate(scores):

#             if score < 0.5:
#                 break
            
#             print(np.int32(cls_dets[det_idx][:-1]))
#             print(labelmap[j-1])
#             Xmin, Xmax = np.int32(cls_dets[det_idx][0]), np.int32(cls_dets[det_idx][2])
#             Ymin, Ymax = np.int32(cls_dets[det_idx][1]), np.int32(cls_dets[det_idx][3])
#             # pts = np.array(((Xmin, Ymin),(Xmin, Ymax),(Xmax, Ymax),(Xmax, Ymin), j))
#             pts = np.array((Xmin, Xmax, Ymin, Ymax, j, score))
#             pts_list.append(pts)
        
#     return np.array(pts_list)

# def object_detect_all_boxes(net, cuda, image, transform, im_size=300, thresh=0.05):
#     # all detections are collected into:
#     #    all_boxes[cls][image] = N x 5 array of detections in
#     #    (x1, y1, x2, y2, score)
#     all_boxes = [[]for _ in range(len(labelmap)+1)]
#     h, w, _ = image.shape

#     # imageのサイズを300に合わせる
#     x, _, _ = transform(image)
#     x = x[:, :, (2, 1, 0)]
#     x = torch.from_numpy(x).permute(2, 0, 1)
#     x = Variable(x.unsqueeze(0))
#     if args.cuda:
#         x = x.cuda()

#     detections = net(x).data

#     # skip j = 0, because it's the background class
#     for j in range(1, detections.size(1)):
#         dets = detections[0, j, :]
#         mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
#         dets = torch.masked_select(dets, mask).view(-1, 5)
#         if dets.size(0) == 0:
#             continue
#         boxes = dets[:, 1:]
#         boxes[:, 0] *= w
#         boxes[:, 2] *= w
#         boxes[:, 1] *= h
#         boxes[:, 3] *= h
#         scores = dets[:, 0].cpu().numpy()
#         cls_dets = np.hstack((boxes.cpu().numpy(),
#                                 scores[:, np.newaxis])).astype(np.float32,
#                                                                 copy=False)
#         all_boxes[j] = cls_dets

#     return all_boxes

# def threshold_bbox(all_boxes, threshold = 0.5):
#     pts_list = []
#     for j in range(1,len(labelmap) + 1):
#         cls_dets = all_boxes[j]
#         scores = cls_dets[:,-1]
#         for det_idx, score in enumerate(scores):
#             if score > 0.3:
#                 print(np.int32(cls_dets[det_idx][:-1]))
#                 print(labelmap[j])
#                 Xmin, Xmax = np.int32(cls_dets[det_idx][0]), np.int32(cls_dets[det_idx][2])
#                 Ymin, Ymax = np.int32(cls_dets[det_idx][1]), np.int32(cls_dets[det_idx][3])
#                 # pts = np.array(((Xmin, Ymin),(Xmin, Ymax),(Xmax, Ymax),(Xmax, Ymin), j))
#                 pts = np.array((Xmin, Xmax, Ymin, Ymax, j))
#                 pts_list.append(pts)

#     return np.array(pts_list), np.array(scores)


