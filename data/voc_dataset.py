# -*- coding: UTF-8 -*-
import os
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import Dataset
from .util import read_img

class VOCDataset(Dataset):

    def __init__(self, data_dir, split="trainval", use_difficult=False, color=True):
        super(VOCDataset, self).__init__()
        self.data_dir = data_dir
        id_list_file = os.path.join(data_dir, "ImageSets/Main", "{0}.txt".format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file, 'r')]
        self.use_difficult= use_difficult
        self.color = color

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        bbox = list()
        label = list()
        difficult = list()

        tree = ET.parse(os.path.join(self.data_dir, "Annotations", "{0}.xml".format(id_)))
        for obj in tree.findall("object"):

            if not self.use_difficult and int(obj.find("difficult").text) == 1:
                continue
            difficult.append(int(obj.find("difficult").text))
            name = obj.find("name").text
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
            bnd_box = obj.find("bndbox")
            bbox.append([int(bnd_box.find(tag).text)
                             for tag in ["xmin", "ymin", "xmax", "ymax"]])

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, np.bool8).astype(np.int8)

        img_file = os.path.join(self.data_dir, "JPEGImages", id_+".jpg")
        img = read_img(img_file)

        return img, bbox


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
