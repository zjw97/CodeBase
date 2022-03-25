import copy
import os
from glob import glob
import cv2
import matplotlib as plt
import numpy as np
from evaluate import *
import torch
from torch import nn
import torch.nn.functional as F
import albumentations as A

from remo import parse_remo_xml

torch.manual_seed(2022)

# transform = A.Compose([
#     A.RandomCropNearBBox(),
# ], bbox_params=A.BboxParams(format="pascal_voc"))

aic_root = "/home/zjw/Datasets/AIC_Data"
aic_xml_path = "/home/zjw/Datasets/AIC_Data/Layout/train_PersonWithFaceHeadHand_V0_V0_cont1_V0_cont2_V0_cont3.txt"

remo_aic_root = "/home/zjw/Datasets/AIC_REMOCapture"
remo_aic_path = "/home/zjw/Datasets/AIC_REMOCapture/XML/REMO_AIC_handface_20180530"

xml_list = []
with open(aic_xml_path, "r") as f:
    for line in f:
        xml_list.append((aic_root, os.path.join(aic_root, line.strip())))

idx = 0
while True:
    root, xml_path = xml_list[idx]
    basename = os.path.basename(xml_path)
    meta = parse_remo_xml(aic_root, xml_path)
    image = cv2.imread(meta["image_path"])
    h, w, _ = image.shape
    boxes = meta["boxes"]
    boxes = np.array(boxes)
    aic_image = copy.deepcopy(image)
    draw_boxes(aic_image, boxes)
    # image = transform(image=image, bboxes=boxes.astype("float"), cropping_bbox=boxes[0])["image"]

    remo_aic_xml_path = os.path.join(remo_aic_path, basename)
    if os.path.exists(remo_aic_xml_path):
        remo_aic_image = copy.deepcopy(image)
        meta = parse_remo_xml(remo_aic_root, remo_aic_xml_path)
        boxes = meta["boxes"]
        boxes = np.array(boxes)
        draw_boxes(remo_aic_image, boxes)
        cv2.imshow("remo_aic_image", remo_aic_image)
        cv2.moveWindow("remo_aic_image", w+100, 0)


    cv2.imshow("AIC", aic_image)
    cv2.moveWindow("AIC", 100, 0)
    key = cv2.waitKey(0)
    if key == ord("q"):
        break
    elif key == ord("a"):
        idx -= 1
    else:
        idx += 1
    cv2.destroyAllWindows()

def register_model(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # add entries to registry dict/sets
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    has_pretrained = False  # check if model has a pretrained url to allow filtering on this
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        # this will catch all models that have entrypoint matching cfg key, but miss any aliasing
        # entrypoints or non-matching combos
        has_pretrained = 'url' in mod.default_cfgs[model_name] and 'http' in mod.default_cfgs[model_name]['url']
        _model_default_cfgs[model_name] = deepcopy(mod.default_cfgs[model_name])
    if has_pretrained:
        _model_has_pretrained.add(model_name)
    return fn



