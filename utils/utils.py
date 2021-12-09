# -*- coding: UTF-8 -*-
import random
import numpy as np
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed(seed)

def generate_anchor_point(feature_scale, step, w, h):
    shift_x = np.arange(0, w * feature_scale, step)
    shift_y = np.arange(0, h * feature_scale, step)

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    anchor_point = np.stack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()], axis=1)

    return anchor_point


def generate_anchor(anchor_scale, aspect_ratio, feature_scale):
    n = len(anchor_scale) * len(aspect_ratio)
    anchors = np.zeros((n , 4))
    i = 0
    for scale in anchor_scale:
        for ratio in aspect_ratio:
            width = feature_scale * scale * np.sqrt(ratio)
            height = feature_scale * scale / np.sqrt(ratio)
            anchors[i, 0] = - width / 2
            anchors[i, 1] = - height / 2
            anchors[i, 2] = width / 2
            anchors[i, 3] = height / 2
            i += 1

    return anchors


def anchor_generator(anchor_scale, aspect_ratio, feature_scale, step, w, h):
    anchor_point = torch.tensor(generate_anchor_point(feature_scale, step, w, h))
    anchors = torch.tensor(generate_anchor(anchor_scale, aspect_ratio, feature_scale))
    N = anchor_point.shape[0]
    K = anchors.shape[0]
    anchor_point = anchor_point.reshape(N, 1, 4).expand(N, K, 4)
    anchors = anchors.reshape(1, K, 4).expand(N, K, 4)
    anchors = torch.add(anchor_point, anchors).reshape(-1, 4)
    return anchors

if __name__ == "__main__":
    anchor_generator([8, 16 ,32], [0.5, 1, 2.], 16, 16, 150, 150)