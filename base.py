import os
import cv2
import random

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from skimage.exposure import adjust_gamma

from remo import parse_remo_xml

transforms = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.5),
])
plt.ion()

if __name__ == "__main__":
    xml_list = []
    data_root = "/home/zjw/Datasets/AIC_REMOCapture"
    with open("/home/zjw/Datasets/AIC_REMOCapture/txt/trainval_AIC_remocap2018053008070827.txt") as f:
        for line in tqdm(f):
            xml_list.append(os.path.join(data_root, line.strip()))

    random.seed(2022)
    random.shuffle(xml_list)

    idx = 0
    while True:
        xml_path = xml_list[idx]
        meta = parse_remo_xml(data_root, xml_path)
        image_path = meta["image_path"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(15, 7.5))
        plt.subplot(121)
        plt.imshow(image)

        boxes = np.array(meta["boxes"])
        plt.subplot(122)
        image = transforms(image=image)["image"]

        plt.imshow(image)
        plt.show()
        # plt.pause(1)
        key = input()
        plt.clf()

        if key == "q":
            exit()
        elif key == "a":
            idx -= 1
        else:
            idx += 1
