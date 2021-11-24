import torch

def nms(bboxes, scores, threshold=0.5):
    """
    最好先移除置信度低于阈值的prior box不然会很慢
    bboxes: <torch.tensor>
    scores: <torch.tensor>
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)  # 计算面积
    _, order = scores.sort(0, descending=True)  # 排序
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        # 计算 box[i] 与其余各框的IOU
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1) * (yy2 - yy1).clamp(0)  # [N-1, ]

        iou = inter / (areas[i] + areas[order[1:]] - inter) # [N-1, ]
        idx = (iou <= threshold).nonzero().squeeze()  # 此时的idx为[N-1], order为[N, ]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间额差值
    return torch.LongTensor(keep)  #  Pytorch的索引值为LongTensor


def calc_iou_tensor(bbox1, bbox2):
    """
    Args:
        Calculation IOU based on two box tensor
        reference to https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/ssd/utils.py
        bbox1:  shape(N, 4)
        bbox2:  shape(M, 4)
    Returns:
        iou: shape(N, M)

    """

    N = bbox1.size(0)
    M = bbox2.size(0)

    be1 = bbox1.unsqueeze(1).expand(-1, M, -1)
    be2 = bbox2.unsqueeze(0).expand(N, -1, -1)

    lt = torch.max(be1[:, :, :2], be2[:, :, :2])  # xmin, ymin
    rb = torch.min(be1[:, :, 2:], be2[:, :, 2:])  # xmax, ymax

    delta = rb - lt  # 直接对应位置相减求得width， height
    delta[delta < 0] = 0

    inter = delta[:, :, 0] * delta[:, :, 1]

    delta1 = be1[:, :, 2:] - be1[:, :, :2]
    area1 = delta1[:, :, 0] * delta[:, :, 1]
    delta2 = be2[:, :, 2:] - be2[:, :, :2]
    area2 = delta2[:, :, 0] * delta2[:, :, 1]

    iou = inter / (area1 + area2 - inter)
    return iou


class AveragerMeter():
    def __init__(self):
        self.val = 0
        self.total = 0
        self.avg = 0
        self.sum = 0

    def update(self, val, n):
        self.val = val
        self.total += n
        self.sum += val * n
        self.avg = self.sum / self.total


