# coding:utf-8

from data import *
from utils.augmentations import SSDTestTransformer
from layers.modules.multibox_loss import MultiBoxLoss_MapClasses
from ssd import build_ssd
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import csv
from gensim.models import KeyedVectors
import nltk
stop_words = nltk.corpus.stopwords.words('english')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='COCO', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default='', 
                    help='Dataset root directory path')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--result_folder', default='result',
                    help='Directory for saving checkpoint models')
parser.add_argument('--pretrained_model', default='trained_model/ssd300_COCO_147.pth',
                    help='Path to pretrained weight')
# scene text parser
parser.add_argument('--wv_model_path', default='wv_models/self_train_200.bin', type=str, help='path to word vector model')
parser.add_argument('--scene_text_data_root', default='/media/kouki/kouki/scene_text_data', type=str, help='path to word vector model')
parser.add_argument('--map_classes', default=False, type=str, help='')
args = parser.parse_args()


# デバッグ用
wv = True 

if wv:
    attn_block_index = [0]
    wv_model_path = os.path.join(args.scene_text_data_root, args.wv_model_path)
    wv_model = KeyedVectors.load_word2vec_format(wv_model_path, binary=True)
else:
    attn_block_index = []
    wv_model = None
# デバッグ用
reject_words = stop_words


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.result_folder):
    os.mkdir(args.result_folder)

def train():
    if attn_block_index:
        wv_model_for_dataset = wv_model
    else:
        wv_model_for_dataset = None

    if args.dataset == 'COCO':
        # if args.dataset_root == VOC_ROOT:
        #     if not os.path.exists(COCO_ROOT):
        #         parser.error('Must specify dataset_root if specifying dataset')
        #     print("WARNING: Using default COCO dataset_root because " +
        #           "--dataset_root was not specified.")
        args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=COCO_ROOT, image_set='train2014',
                                transform=SSDTestTransformer(cfg['min_dim'], mean=MEANS),
                                wv_model=wv_model_for_dataset, reject_words=reject_words)

    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDTestTransformer(cfg['min_dim'], mean=MEANS))
                               
    if args.dataset == 'OpenImages':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(OpenImages_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = OpenImages_ROOT
        cfg = openimages
        dataset = OpenImagesDetection(root=OpenImages_ROOT, image_set='train',
                                transform=SSDTestTransformer(cfg['min_dim'], mean=MEANS),
                                wv_model=wv_model_for_dataset, reject_words=reject_words, 
                                cats=OPENIMAGES_CATS)


    train_size = int(len(dataset) * 0.9)
    num_samples = len(dataset)
    dataset_train = Subset(dataset, range(train_size))
    dataset_val = Subset(dataset, range(train_size, num_samples))

    if args.map_classes:
        ssd_net = build_ssd('test', cfg['min_dim'], wv_model.vector_size, dataset_train.dataset.cats,
                            wv_model=wv_model, attn_blocks_index=attn_block_index, 
                            dataset=args.dataset, map_classes=args.map_classes, use_cuda=args.cuda)
        criterion = MultiBoxLoss_MapClasses(0.5, True, 0, True, 3, 0.5, False, 
                                                    args.cuda, ssd_net.class_vectors)
    else:
        ssd_net = build_ssd('test', cfg['min_dim'], len(dataset_train.dataset.cats) + 1,
                            dataset_train.dataset.cats, wv_model=wv_model, 
                            attn_blocks_index=attn_block_index, 
                            dataset=args.dataset, use_cuda=args.cuda)



    if args.pretrained_model:
        print('Loading weight at {}...'.format(args.pretrained_model))
        ssd_net.load_weights(args.pretrained_model)

    if args.cuda:
        ssd_net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
        ssd_net.to(device)

    collate = Collate(dataset.wv_model is not None, True)
    data_loader_val = data.DataLoader(dataset_val, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=collate.detection_collate,
                                  pin_memory=True)
    
    ssd_net.eval()


#############################
    from matplotlib import pyplot as plt
    from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
    import glob
    import cv2

    with torch.no_grad():
        for batch in data_loader_val:
            imgs, targets, word_embs, img_ids = batch
            # img_name = dataset_val.dataset.coco.loadImgs(img_id)[0]['file_name']
            top_k=10
            y, weights = ssd_net(imgs, word_embs)
            detections = y.data
            

            for det, img_id in zip(detections, img_ids):
                img_name = dataset_val.dataset.coco.loadImgs(img_id)[0]['file_name']
                img_path = os.path.join(args.dataset_root, 'images', 'train2014', img_name)
                image = cv2.imread(img_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    
                plt.figure(figsize=(10,10))
                plt.imshow(rgb_image)  # plot the image for matplotlib
                currentAxis = plt.gca()
                id2cat = {v:k for k, v in dataset_val.dataset.label_map.items()}
    
                for i in range(det.size(0)):
                    j = 0
                    while det[i,j,0] >= 0.4:
                        score = det[i,j,0]
                        display_txt = '%s: %.2f'%(id2cat[i], score)
                        pt = (det[i,j,1:]*scale).cpu().numpy()
                        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                        color = plt.get_cmap("tab10")(i+2)
                        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
                        j+=1
                plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    # here we specify year (07 or 12) and dataset ('test', 'val', 'train') 
    for path in glob.glob('sample*'):
        image = cv2.imread(path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # View the sampled input image before transform
        # plt.figure(figsize=(10,10))
        # plt.imshow(rgb_image)
        # plt.show()

        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1).cuda().unsqueeze(0)
        y, _ = ssd_net(x, None)
        top_k=10

        plt.figure(figsize=(10,10))
        plt.imshow(rgb_image)  # plot the image for matplotlib
        currentAxis = plt.gca()

        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        id2cat = {v:k for k, v in dataset_val.dataset.label_map.items()}
        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= 0.4:
                score = detections[0,i,j,0]
                display_txt = '%s: %.2f'%(id2cat[i], score)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                color = plt.get_cmap("tab10")(i+2)
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
                j+=1
        plt.show()

#############################







            

class Result:
    def __init__(self, label_map):
        self.xmin = []
        self.ymin = []
        self.xmax = []
        self.ymax = []
        self.labels = []
        self.label_map = label_map
        
    def update(self, bboxes, labels):
        self.xmin += bboxes[:, 0]
        self.ymin += bboxes[:, 1]
        self.xmax += bboxes[:, 2]
        self.ymax += bboxes[:, 3]
        
        self.labels.append()
        self.label_map.append

if __name__ == '__main__':
    train()
