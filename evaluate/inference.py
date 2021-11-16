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