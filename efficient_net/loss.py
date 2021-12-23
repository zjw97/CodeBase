import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class SmoothedEntropyLoss(_WeightedLoss):
    # https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/173733
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss