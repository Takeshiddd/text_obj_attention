# coding:utf-8

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules.multibox_loss import MultiBoxLoss, MultiBoxLoss_MapClasses
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import csv
from gensim.models import KeyedVectors
import nltk
stop_words = nltk.corpus.stopwords.words('english')
from torch.optim.lr_scheduler import LambdaLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='COCO', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=2, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=False, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--pretrained_model', default='trained_model_map_classes_collect/ssd300_COCO_100.pth', type=str,
                    help='Checkpoint state_dict file to map classed pretrained weights')
                    
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.4, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
# scene text parser
parser.add_argument('--wv_model_path', default='wv_models/self_train_200.bin', type=str, help='path to word vector model')
parser.add_argument('--scene_text_data_root', default='/media/kouki/kouki/scene_text_data', type=str, help='path to word vector model')
parser.add_argument('--map_classes', default=False, type=str, help='')
args = parser.parse_args()


# デバッグ用
wv = True 
cats = ['car', 'stop sign']
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

def train():
    if attn_block_index:
        wv_model_for_dataset = wv_model
    else:
        wv_model_for_dataset = None

    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=COCO_ROOT, image_set='train2014',
                                transform=SSDAugmentation(cfg['min_dim'], mean=MEANS),
                                wv_model=wv_model_for_dataset, reject_words=reject_words, 
                                cats=cats)

    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'], mean=MEANS))
                               
    if args.dataset == 'OpenImages':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(OpenImages_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = OpenImages_ROOT
        cfg = openimages
        dataset = OpenImagesDetection(root=OpenImages_ROOT, image_set='train',
                                transform=SSDAugmentation(cfg['min_dim'], mean=MEANS),
                                wv_model=wv_model_for_dataset, reject_words=reject_words, 
                                cats=OPENIMAGES_CATS)


    train_size = int(len(dataset) * 0.9)
    num_samples = len(dataset)
    dataset_train = Subset(dataset, range(train_size))
    dataset_val = Subset(dataset, range(train_size, num_samples))
    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    
    if args.map_classes:
        ssd_net = build_ssd('train', cfg['min_dim'], wv_model.vector_size, dataset_train.dataset.cats,
                            wv_model=wv_model, attn_blocks_index=attn_block_index, 
                            dataset=args.dataset, map_classes=args.map_classes, use_cuda=args.cuda)
        criterion = MultiBoxLoss_MapClasses(0.5, True, 0, True, 3, 0.5, False, args.cuda, ssd_net.class_vectors)
    else:
        ssd_net = build_ssd('train', cfg['min_dim'], len(dataset_train.dataset.cats) + 1,
                            dataset_train.dataset.cats, wv_model=wv_model, 
                            attn_blocks_index=attn_block_index, 
                            dataset=args.dataset, use_cuda=args.cuda)
        criterion = MultiBoxLoss(dataset_train.dataset.cats, 0.5, True, 0, 
                             True, 3, 0.5, False, args.cuda)

    if args.pretrained_model:
        print('Loading pretrained model {}...'.format(args.pretrained_model))
        ssd_net.load_pretrained_weights(args.pretrained_model)
    elif args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    if args.cuda:
        ssd_net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
        ssd_net.to(device)

    optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: 0.98 ** epoch)
    
    ssd_net.train()
    # loss counters
    step_index = 0
    iteration = 0
    val_data_num = len(dataset_val)

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss (train)', 'Conf Loss (train)', 'Total Loss (train)', 
                            'Loc Loss (val)', 'Conf Loss (val)', 'Total Loss (val)']

    print('Loading the dataset...')
    print('Batch size:' + str(args.batch_size))
    collate = Collate(dataset.wv_model is not None)
    data_loader_train = data.DataLoader(dataset_train, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=collate.detection_collate,
                                  pin_memory=True)
    data_loader_val = data.DataLoader(dataset_val, args.batch_size // 2,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=collate.detection_collate,
                                  pin_memory=True)
    
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')

    for epoch in range(500):
        # reset epoch loss counters
        loc_loss = 0
        conf_loss = 0
        loc_loss_val = 0
        conf_loss_val = 0
        # create batch iterator
        batch_iterator = iter(data_loader_train)
        for batch in batch_iterator:
            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            # load train data
            images, targets, word_emb_img = batch

            if args.cuda:
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]
                if word_emb_img is not None:
                    word_emb_img = word_emb_img.to(device)
            
            # train forward
            t0 = time.time()
            out, _ = ssd_net(images, word_emb_img)
            # train backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            loc_loss += loss_l.data
            conf_loss += loss_c.data
            t1 = time.time()
            if iteration % 20 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('[epoch ' + str(epoch) + '] iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data) + ' '
                            + '||loss_l: %.4f ||' % (loss_l.data) + ' ' + '||loss_c: %.4f ||' % (loss_c.data) + ' ')
            iteration += 1      
        
        if args.visdom:
            print('timer: %.4f sec.' % (t1 - t0))
            print('[epoch ' + str(epoch) + '] iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data) + ' '
                        + '||loss_l: %.4f ||' % (loss_l.data) + ' ' + '||loss_c: %.4f ||' % (loss_c.data) + ' ')
            
            ssd_net.eval()
            with torch.no_grad():
                batch_iterator_val = iter(data_loader_val)
                for batch_val in tqdm(batch_iterator_val):
                    images_val, targets_val, word_emb_img_val = batch_val
                    if args.cuda:
                        images_val = images_val.to(device)
                        targets_val = [(ann.to(device)) for ann in targets_val]
                    out_val, _ = ssd_net(images_val, word_emb_img_val)
                    optimizer.zero_grad()
                    # val loss
                    loss_l_val, loss_c_val = criterion(out_val, targets_val)
                    loc_loss_val += loss_l_val.data
                    conf_loss_val += loss_c_val.data

            ssd_net.train()
                    
            if epoch == 0:
                iter_plot = create_vis_plot(viz, loc_loss, conf_loss, loc_loss_val, conf_loss_val, 'Epoch', 'Loss',
                                vis_title, vis_legend, args.batch_size, val_data_num)
            else:
                # val forward
                update_vis_plot(viz, epoch, loc_loss, conf_loss, loc_loss_val, conf_loss_val,
                            iter_plot, 'append', args.batch_size, val_data_num)
        
        scheduler.step()

        if not os.path.isdir("Log"):
            os.mkdir("Log")  
        with open('Log/ssd300_COCO.csv', 'w') as f:
            writer = csv.writer(f)
            row = [epoch, loc_loss / int(31666/16), conf_loss / int(31666/16), (loc_loss + conf_loss) / int(31666/16),  loc_loss_val / int(3159/16), conf_loss_val / int(3159/16), (loc_loss_val + conf_loss_val) / int(3159/16)]
            writer.writerow(row)

        print('Saving state, epoch:', epoch)

        model_name = 'model_swe_pretrain-classes:stop_car'
        save_dir = os.path.join('trained_model', model_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(ssd_net.module.state_dict(), os.path.join(save_dir, 'ssd300_COCO_' +
                       repr(epoch) + '.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(viz, loc_train, conf_train, loc_val, conf_val, _xlabel, _ylabel,
                     _title, _legend, _batch_num, _val_data_num):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.Tensor([loc_train/ int(31666/8), conf_train/ int(31666/8), (loc_train + conf_train)/ int(31666/8),
                             loc_val/ int(3159/16), conf_val/ int(3159/8), (loc_val + conf_val)/ int(3159/8)]).unsqueeze(0).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(viz, iteration, loc_train, conf_train, loc_val, conf_val, window, update_type,
                    _batch_num, _val_data_num):
    viz.line(
        X=torch.ones((1, 6)).cpu() * iteration,
        Y=torch.Tensor([loc_train/ int(31666/8), conf_train/ int(31666/8), (loc_train + conf_train)/ int(31666/8),
                             loc_val/ int(3159/8), conf_val/ int(3159/8), (loc_val + conf_val)/ int(3159/8)]).unsqueeze(0).cpu(),
        win=window,
        update=update_type
    )

if __name__ == '__main__':
    train()
