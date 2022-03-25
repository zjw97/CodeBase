import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss


def cross_entropy_smooth(pred, target, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    smoothing = 0.05
    confidence = 1 - smoothing
    class_num = pred.size(1)
    #print('\n class_num --- \n', class_num)
    #print('-- use_softmax cross_entropy_smooth label ======= \n ---- size:  ', target.size())
    pred = pred.log_softmax(dim=-1)
    #print('pred : ',pred.size(), "  ",pred)
    #print('weight : ',weight.unsqueeze(1).size(), "  ",weight)
    if weight is not None : 
        pred = pred * weight.unsqueeze(1)   
    with torch.no_grad():
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(smoothing/ (class_num -1))
        true_dist.scatter_(1, target.data.unsqueeze(1),confidence)
    #print('true_dist : ',true_dist.size(), "  ",true_dist)
    loss = torch.mean(torch.sum(-true_dist * pred , dim = -1 ))

    return loss


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy_smooth(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
    #print(label[label>0])
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
        
    #print('-- use_sigmoid binary_cross_entropy_smooth label =======\n ---- size:  ', label.size())
    #print('--- pred size = ' , pred.size())
    label_f =label.float()
    label_smooth = (label_f * (1 - 0.1) + 0.1 /2).float()
    #label_print =  label_smooth.view(-1)
    #print('smooth----- ',label_print[label_print>0.05])
    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label_smooth, weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy_smooth(pred, target, label, reduction='mean', avg_factor=None):
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    print(label)
    print('-- use_mask mask_cross_entropy_smooth label ======= \n ---- size:  ', label.size())
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]

@LOSSES.register_module
class CrossEntropyLoss_Smooth(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(CrossEntropyLoss_Smooth, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy_smooth
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy_smooth
        else:
            self.cls_criterion = cross_entropy_smooth

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        print('self.use_sigmoid = ', self.use_sigmoid)
        print('self.loss_weight = ', self.loss_weight)
        print('cls_score = ', cls_score.size())
        print('label = ', label.size())
        print('weight = ', weight)
        print('avg_factor = ', avg_factor)
        print('kwargs = ', **kwargs)
        print('reduction = ', reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
