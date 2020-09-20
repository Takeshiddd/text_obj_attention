# coding:utf-8

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
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
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=False, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.4, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=COCO_ROOT, image_set='train2014',
                                transform=SSDAugmentation(cfg['min_dim'], MEANS))


    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'], MEANS))

    train_size = int(len(dataset) * 0.9)
    num_samples = len(dataset)
    dataset_train = Subset(dataset, range(train_size))
    dataset_val = Subset(dataset, range(train_size, num_samples))
    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', WV_MODEL, cfg['min_dim'], cfg['num_classes'])

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    if args.cuda:
        ssd_net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
        ssd_net = ssd_net.cuda()

        
    optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
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
    data_loader_train = data.DataLoader(dataset_train, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    data_loader_val = data.DataLoader(dataset_val, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate,
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
            images, targets = batch

            with torch.no_grad():
                if args.cuda:
                    images = Variable(images.cuda())
                    targets = [Variable(ann.cuda()) for ann in targets]
                else:
                    images = Variable(images)
                    targets = [Variable(ann) for ann in targets]
            
            # train forward
            t0 = time.time()
            out = ssd_net(images)
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
            batch_iterator_val = iter(data_loader_val)
            for batch_val in tqdm(batch_iterator_val):
                images_val, targets_val = batch_val
                with torch.no_grad():
                    if args.cuda:
                        images_val = Variable(images_val.cuda())
                        targets_val = [Variable(ann.cuda()) for ann in targets_val]
                    else:
                        images_val = Variable(images_val)
                        targets_val = [Variable(ann) for ann in targets_val]
                out_val = ssd_net(images_val)
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
            
        with open('Log/ssd300_COCO.csv', 'w') as f:
            writer = csv.writer(f)
            row = [epoch, loc_loss / int(31666/16), conf_loss / int(31666/16), (loc_loss + conf_loss) / int(31666/16),  loc_loss_val / int(3159/16), conf_loss_val / int(3159/16), (loc_loss_val + conf_loss_val) / int(3159/16)]
            writer.writerow(row)

        print('Saving state, epoch:', epoch)

        torch.save(ssd_net.state_dict(), 'weights_contenue/ssd300_COCO_' +
                       repr(epoch + 410) + '.pth')


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
