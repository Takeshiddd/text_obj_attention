import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from data import voc, coco, openimages
import os
import re
import numpy as np
from gensim.models import word2vec


#############################################################################################################
############################################ edited by takeshita ############################################
class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes, 
            attn_blocks_index=[], weight_func='cos_sim', class_vectors=None, 
            weight_activation='relu', word_cnn=None, dataset='COCO'):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        if dataset=='COCO': 
            self.cfg = coco 
        elif dataset=='VOC':
            self.cfg = voc
        elif dataset=='OpenImages':
            self.cfg = openimages  
        else:
            print('invalid dataset')
            raise ValueError
        self.priorbox = PriorBox(self.cfg)
        # self.priors = Variable(self.priorbox.forward(), volatile=True)
        with torch.no_grad():
            self.priors = torch.tensor(self.priorbox.forward())
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.class_vectors = class_vectors
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            num_classes = self.num_classes if self.class_vectors is None else self.class_vectors.shape[0]
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45,
                                        class_vectors=self.class_vectors)

        self.attn_blocks_index = sorted(attn_blocks_index)
        self.word_cnn = word_cnn

        self.attn_blocks = []
        for _ in self.attn_blocks_index:
            self.attn_blocks.append(Attn(weight_func, weight_activation))

    def forward(self, x, word_emb_img):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        if self.attn_blocks_index:
            values = self.word_cnn(word_emb_img)
            
        sources = list()
        loc = list()
        conf = list()
        weights = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
            # aply attention 
            if k == 15 and 0 in self.attn_blocks_index:
                idx = self.attn_blocks_index.index(0)
                attn_block = self.attn_blocks[idx]
                value = values[idx]
                x, weight = attn_block.attn_func(x, value)
                weights.append(weight)

            if k == 22 and 1 in self.attn_blocks_index:
                idx = self.attn_blocks_index.index(1)
                attn_block = self.attn_blocks[idx]
                value = values[idx]
                x, weight = attn_block.attn_func(x, value)
                weights.append(weight)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
            # aply attention
            if k == 29 and 2 in self.attn_blocks_index:
                idx = self.attn_blocks_index.index(2)
                attn_block = self.attn_blocks[idx]
                value = values[idx]
                x, weight = attn_block.attn_func(x, value)
                weights.append(weight)

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                conf.view(conf.size(0), -1, self.num_classes),                # conf preds
                self.priors.type(type(x.data)))
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output, weights

    def load_weights(self, base_file):
        _, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(fix_model_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage)))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def load_pretrained_weights(self, base_file):
        _, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            state_dict = fix_model_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            for key in list(state_dict.keys()):
                if key.split('.')[0] == 'conf':
                    del state_dict[key]
            self.load_state_dict(state_dict, strict=False)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

class WordCNN(nn.Module):
    def __init__(self, base_cfg, image_size=300, wv_dim=200, attn_blocks_index=[]):
        super(WordCNN, self).__init__()
        self.image_size = image_size
        self.wv_dim = wv_dim 
        self.attn_blocks_index = sorted(attn_blocks_index)
        self.attn_blocks = []
        
        if len(self.attn_blocks_index) > 0:
            block0 = [nn.Conv2d(self.wv_dim, self.wv_dim, kernel_size=3, padding=1),
                      nn.Conv2d(self.wv_dim, self.wv_dim, kernel_size=3, padding=1),
                      get_pooling_layer(base_cfg[2]),
                      nn.Conv2d(self.wv_dim, self.wv_dim, kernel_size=3, padding=1),
                      nn.Conv2d(self.wv_dim, self.wv_dim, kernel_size=3, padding=1),
                      get_pooling_layer(base_cfg[5]),
                      nn.Conv2d(self.wv_dim, base_cfg[6], kernel_size=3, padding=1),
                      nn.Conv2d(base_cfg[6], base_cfg[7], kernel_size=3, padding=1),
                      nn.Conv2d(base_cfg[7], base_cfg[8], kernel_size=3, padding=1)]
            self.block0 = nn.ModuleList(block0)
            self.attn_blocks.append(self.block0)

        if max(self.attn_blocks_index) >= 1:
            block1 = [get_pooling_layer(base_cfg[9]),
                      nn.Conv2d(base_cfg[8], base_cfg[10], kernel_size=3, padding=1),
                      nn.Conv2d(base_cfg[10], base_cfg[11], kernel_size=3, padding=1),
                      nn.Conv2d(base_cfg[11], base_cfg[12], kernel_size=3, padding=1)]
            self.block1 = nn.ModuleList(block1)
            self.attn_blocks.append(self.block1)
        
        if max(self.attn_blocks_index) >= 2:
            block2 = [get_pooling_layer(base_cfg[13]),
                      nn.Conv2d(base_cfg[12], base_cfg[14], kernel_size=3, padding=1),
                      nn.Conv2d(base_cfg[14], base_cfg[15], kernel_size=3, padding=1),
                      nn.Conv2d(base_cfg[15], base_cfg[16], kernel_size=3, padding=1)]
            self.block2 = nn.ModuleList(block2)
            self.attn_blocks.append(self.block2)
        
        if max(self.attn_blocks_index) >= 3:
            print('max attn_layer number should be set 2 or less')
            raise ValueError

    def forward(self, x):
        values = []
        for i, block in enumerate(self.attn_blocks):
            for b in block:
                x = b(x)
            if i in self.attn_blocks_index:
                values.append(x)
        return values


class Attn(object):
    def __init__(self, weight_func='cos_sim', weight_activation='relu'):
        self.weight_func = weight_func
        self.weight_activation = weight_activation

        if self.weight_func == 'cos_sim' or self.weight_func == 'dot':
            self._weight_func = self._weight_function

        if not self.weight_activation:
            self._weight_activation = lambda x: x
        elif self.weight_activation == 'relu':
            self._weight_activation = nn.ReLU(inplace=True)

    def attn_func(self, image_feature, word_feature):
        weights = self._weight_function(image_feature, word_feature)
        weighted_word_feature = weights.unsqueeze(3) * word_feature.unsqueeze(1).unsqueeze(1)
        mixed_feature = image_feature + weighted_word_feature.sum(dim=(4,5)).permute(0,3,1,2)
        return mixed_feature.contiguous(), weights

    def _weight_function(self, image_feature, word_feature):
        batch_size, channel_size, height, width = image_feature.size()
        _, _, height_w, width_w = word_feature.size()
        image_feature_disamb = image_feature.permute(0,2,3,1).view(batch_size, -1, channel_size)
        word_feature_disamb = word_feature.permute(0,2,3,1).view(batch_size, -1, channel_size)
        
        if self.weight_func == 'cos_sim':
            norm = torch.norm(image_feature_disamb, dim=2).unsqueeze(-1)
            # norm[norm == 0] = 1
            image_feature_disamb = image_feature_disamb / norm  # Don't set the operator as /= in order not to change original data of image feature.
            norm = torch.norm(word_feature_disamb, dim=2).unsqueeze(-1) #
            # norm[norm == 0] = 1
            word_feature_disamb = word_feature_disamb / norm  # Don't set the operator as /= in order not to change original data of image feature.

        mat_prod = torch.bmm(image_feature_disamb, word_feature_disamb.permute(0,2,1))

        weights = mat_prod.view(batch_size, height, width, height_w, width_w)
        weights = self._weight_activation(weights)
        return weights
    


def get_pooling_layer(v):
    if v == 'M':
        layer = nn.MaxPool2d(kernel_size=2, stride=2)
    elif v == 'C':
        layer = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    return layer

############################################ maked by takeshita ############################################
#############################################################################################################
def fix_model_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False, attn_blocks_index=[]):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}

def get_class_vec(classes, wv_model, bg_vector='cartesian', norm_vec=True):
    class_vec = class_vec_(classes, wv_model, norm_vec)
    if bg_vector == 'zero':
        bg = np.zeros(wv_model.vector_size)
    elif bg_vector == 'cartesian':
        eig_val, eig_vec = get_carte_prod_(class_vec)
        print('maximum eiginvalue: {:.6e}'.format(eig_val[0]))
        print('minimum eiginvalue: {:.6e}'.format(eig_val[-1]))
        bg = eig_vec[:, -1].astype(np.float64)
    else:
        raise ValueError
    if norm_vec:
        norm = np.linalg.norm(bg)
        if norm == 0:
            norm = 1
        bg = bg / norm
    class_vectors = torch.tensor([bg] + class_vec, 
        requires_grad=False)
    return class_vectors.float()

def class_vec_(classes, wv_model, norm_vec=True):
    class_list = [re.split('[- ]', name) for name in classes]
    vec_list = []
    for classes in class_list:
        vec = np.array([wv_model[word.lower()] for word in classes]).sum(0)
        if norm_vec:
            vec = vec / np.linalg.norm(vec)
        vec_list.append(vec)
    return vec_list

def get_carte_prod_(class_vec):
    class_vec = np.array(class_vec)  # (num_classes, vec_dim)
    class_vec_t = class_vec.transpose()
    sigma = np.dot(class_vec_t, class_vec)
    return np.linalg.eig(sigma)

def build_ssd(phase, size, num_classes, classes, wv_model=None, attn_blocks_index=[],
              dataset='COCO', map_classes=False, use_cuda=False):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3, attn_blocks_index=attn_blocks_index),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    if attn_blocks_index:
        word_cnn = WordCNN(base[str(size)], size, wv_model.vector_size, attn_blocks_index)
    else:
        word_cnn = None
    
    if map_classes:
        class_vectors = get_class_vec(classes, wv_model)
        if use_cuda: 
            class_vectors = class_vectors.to(device)
    else:
        class_vectors = None
        
    return SSD(phase, size, base_, extras_, head_, num_classes, 
            attn_blocks_index=attn_blocks_index, word_cnn=word_cnn,
            dataset=dataset, class_vectors=class_vectors)
