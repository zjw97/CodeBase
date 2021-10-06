# -*- coding: UTF-8 -*-
import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

# read PIL image
def read_img(img_path, dtype, color=True):

    f = Image.open(img_path)
    try:
        if color:
            img = f.convert("RGB")
        else:
            img = f.convert("P")
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, "close"):
            f.close()

    if img.ndim == 2:
        return img[np.newaxis]
    else:
        return img.transpose((2, 0, 1))
