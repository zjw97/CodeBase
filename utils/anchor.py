import numpy as np

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)  # 保存9 * 4种anchor
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])  # 16 * [8, 16, 32]    对应三种anchor size为128, 256, 512的anchor框
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])  # 16 * (8 or 16 or 32) * sqrt(1 or 0.5 or 2)

            index = i * len(anchor_scales) + j  #
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor