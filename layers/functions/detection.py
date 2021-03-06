import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from data import voc as cfg
import torch.nn as nn
import torch.nn.functional as F


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, class_vectors=None):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
        self.class_vectors = class_vectors
        if self.class_vectors is None:
            self.conf_func = nn.Softmax(dim=-1)
        else:
            self.conf_func = self.conf_func_map_classes

    # def forward(self, loc_data, conf_data, prior_data):
        
    #     """
    #     Args:
    #         loc_data: (tensor) Loc preds from loc layers
    #             Shape: [batch,num_priors*4]
    #         conf_data: (tensor) Shape: Conf preds from conf layers
    #             Shape: [batch*num_priors,num_classes]
    #         prior_data: (tensor) Prior boxes and variances from priorbox layers
    #             Shape: [1,num_priors,4]
    #     """
    #     num = loc_data.size(0)  # batch size
    #     num_priors = prior_data.size(0)
    #     output = torch.zeros(num, self.num_classes, self.top_k, 5)
    #     conf_preds = self.conf_func(conf_data).view(num, num_priors,
    #                                 self.num_classes).transpose(2, 1)

    #     outputs = []
    #     # Decode predictions into bboxes.
    #     for i in range(num):
    #         decoded_boxes = decode(loc_data[i], prior_data, self.variance)
    #         # For each class, perform nms
    #         conf_scores = conf_preds[i].clone()
    #         mask = conf_scores[1:, :].max(0)[0].gt(self.conf_thresh)
    #         scores = conf_scores[:, mask]
    #         boxes = decoded_boxes[mask, :]
    #         ids, count = nms(boxes, scores[1:, :].max(0)[0], self.nms_thresh, self.top_k)
    #         output = scores[:, ids[:count]].t(), boxes[ids[:count]]
    #         outputs.append(output)
    #     return outputs
    
    def vec2score(self, conf):
        conf_classification = (conf.unsqueeze(-2) * self.class_vectors.unsqueeze(0)).sum(-1)
        return conf_classification

    def conf_func_map_classes(self, conf_data):
        conf_classification = self.vec2score(conf_data)
        preds = F.softmax(conf_classification, dim=-1)
        return preds




    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = self.conf_func(conf_data).view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # クラスiのscoreがthreash以上のボックスを抽出するmask
                scores = conf_scores[cl][c_mask]  # クラスiのscoreが高かったスコア
                if scores.size(0) == 0:  
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes) 
                boxes = decoded_boxes[l_mask].view(-1, 4) # クラスiのscoreが高かったBox
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        # flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        flt[(rank > self.top_k).unsqueeze(-1).expand_as(flt)] = 0  # fixed refering to https://github.com/amdegroot/ssd.pytorch/issues/168
        return output










############################# back up ##################################
# import torch
# from torch.autograd import Function
# from ..box_utils import decode, nms
# from data import voc as cfg



# class Detect(Function):
#     """At test time, Detect is the final layer of SSD.  Decode location preds,
#     apply non-maximum suppression to location predictions based on conf
#     scores and threshold to a top_k number of output predictions for both
#     confidence score and locations.
#     """
#     def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
#         self.num_classes = num_classes
#         self.background_label = bkg_label
#         self.top_k = top_k
#         # Parameters used in nms.
#         self.nms_thresh = nms_thresh
#         if nms_thresh <= 0:
#             raise ValueError('nms_threshold must be non negative.')
#         self.conf_thresh = conf_thresh
#         self.variance = cfg['variance']

#     def forward(self, loc_data, conf_data, prior_data):
#         """
#         Args:
#             loc_data: (tensor) Loc preds from loc layers
#                 Shape: [batch,num_priors*4]
#             conf_data: (tensor) Shape: Conf preds from conf layers
#                 Shape: [batch*num_priors,num_classes]
#             prior_data: (tensor) Prior boxes and variances from priorbox layers
#                 Shape: [1,num_priors,4]
#         """
#         num = loc_data.size(0)  # batch size
#         num_priors = prior_data.size(0)
#         output = torch.zeros(num, self.num_classes, self.top_k, 5)
#         conf_preds = conf_data.view(num, num_priors,
#                                     self.num_classes).transpose(2, 1)

#         # Decode predictions into bboxes.
#         for i in range(num):
#             decoded_boxes = decode(loc_data[i], prior_data, self.variance)
#             # For each class, perform nms
#             conf_scores = conf_preds[i].clone()
#             for cl in range(1, self.num_classes):
#                 c_mask = conf_scores[cl].gt(self.conf_thresh)
#                 scores = conf_scores[cl][c_mask]
#                 if scores.size(0) == 0:
#                     continue
#                 l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
#                 boxes = decoded_boxes[l_mask].view(-1, 4)
#                 # idx of highest scoring and non-overlapping boxes per class
#                 ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
#                 output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
#                                boxes[ids[:count]]), 1)
#         flt = output.contiguous().view(num, -1, 5)
#         _, idx = flt[:, :, 0].sort(1, descending=True)
#         _, rank = idx.sort(1)
#         flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
#         return output
