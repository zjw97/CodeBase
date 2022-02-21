# -*- coding: UTF-8 -*-
import numpy as np
from PIL import Image
import random

__all__ = ["read_img"]

def read_img(img_file, dtype=np.float32, color=True):

    f = Image.open(img_file) # 返回的是一个fp类型
    try:
        if color:
            img = f.convert("RGB")
        else :
            img = f.convert("P")
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, "close"):
            f.close()

    if img.ndim == 2:
        return img[np.newaxis]
    else: # transpose H, W, C -> C,H, W
        return img.transpose(2, 0, 1)
