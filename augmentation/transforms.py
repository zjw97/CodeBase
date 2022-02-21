# -*- coding: UTF-8 -*-
import copy
import os
import cv2
import random
import numpy as np
from remo.data import parse_remo_xml
from evaluate.visualize import draw_rectangle
import albumentations as A
import easing_functions as ef

class CutOut:
    """
        object detection cutout
    """

    def __init__(self):
        self.min_scale = 0.2
        self.max_scale = 0.5
        self.max_trials = 100

    def __call__(self, img, boxes):
        img_height, img_width, _ = img.shape
        for i in range(self.max_trials):
            scale = random.uniform(self.min_scale, self.max_scale)
            # cut区域的宽和高
            cut_width = int(img_width * scale)
            cut_height = int(img_height * scale)
            # 开始擦除
            cut_xmin = random.randint(0, img_width - cut_width)
            cut_ymin = random.randint(0, img_height - cut_height)
            cut_xmax = cut_xmin + cut_width
            cut_ymax = cut_ymin + cut_height

            cut_box = [cut_xmin, cut_ymin, cut_xmax, cut_ymax]

            satisify = True
            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax, cid = boxes[i][:]
                if cid != 1:
                    continue
                box = [xmin, ymin, xmax, ymax]
                satisify = self.is_satisify(box, cut_box)
                if not satisify:
                    break
            if not satisify:
                continue
            mask = np.ones((img_height, img_width, 3))
            mask[cut_ymin: cut_ymax, cut_xmin: cut_xmax, :] = 0
            img = img * mask
            break

        return img.astype("uint8")

    def is_satisify(self, gt_box, cut_box):

        inter_xmin = max(gt_box[0], cut_box[0])
        inter_ymin = max(gt_box[1], cut_box[1])
        inter_xmax = min(gt_box[2], cut_box[2])
        inter_ymax = min(gt_box[3], cut_box[3])

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax * inter_ymin)

        if inter_area > 0:
            return False
        else:
            return True


class RandomBrightness:
    # 随机亮度变换
    def __init__(self):
        self.brightness_prob = 1
        self.brightness_delta = 20

    def __call__(self, image):
        prob = random.uniform(0, 1)
        if prob > self.brightness_prob:
            return image
        delta = random.uniform(-self.brightness_delta, self.brightness_delta)
        table = np.array([i + delta for i in range(256)])  # uint8会把溢出的部分从0开始重新计算
        image = cv2.LUT(image, table)
        image = np.clip(image, 0, 255).astype("uint8")
        return image

class RandomContrast:
    # 随机对比度拉伸
    def __init__(self):
        self.contrast_prob = 1
        self.lower = 0.5
        self.upper = 1.5
        assert self.upper > self.lower

    def __call__(self, image):
        prob = random.uniform(0, 1)
        if prob > self.contrast_prob:
            return image
        delta = random.uniform(self.lower, self.upper)
        table = np.array([i * delta for i in range(256)])
        image = cv2.LUT(image, table)
        image = np.clip(image, 0, 255).astype("uint8")
        return image



class MotionAffine():

    def __init__(self):
        self.transform = A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10,
                                       interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0,
                                       mask_value=None, always_apply=False, p=1)
        self.seq_length = 15

    def lerp(self, a, b, percentage):
        return a * (1 - percentage) + b * percentage

    def __call__(self, img, label):
        h, w, _ = img.shape
        imgs = [img] * self.seq_length
        labels = [label] * self.seq_length

        configA = self.transform.get_params()

        print("angleA: ", configA["angle"], "transXA: ", configA["dx"], "transYA: ", configA["dy"],
              "scaleA: ", configA["scale"])

        configB = self.transform.get_params()

        print("angleB: ", configB["angle"], "transXB: ", configB["dx"], "transYB: ", configB["dy"],
              "scaleB: ", configB["scale"])

        easing = easing_functions.LinearInOut()
        for i in range(self.seq_length):
            percentage = easing(i / (self.seq_length - 1))
            angle = self.lerp(configA["angle"], configB["angle"], percentage)
            transX = self.lerp(configA["dx"], configB["dx"], percentage)
            transY = self.lerp(configA["dy"], configB["dx"], percentage)
            scale = self.lerp(configA["scale"], configB["scale"], percentage)

            config = {
                "angle": angle,
                "scale": scale,
                "dx": 0, # transX,
                "dy": 0, # transY,
                "cols": w,
                "rows": h,
            }

            print("angle: ", angle, "transX: ", transX, "transY: ", transY, "scale: ", scale)

            imgs[i] = self.transform.apply(imgs[i], **config)
            for j in range(len(labels[i])):
                xmin, ymin, xmax, ymax, cid = labels[i][j][:]
                xmin = xmin / w
                ymin = ymin / h
                xmax = xmax / w
                ymax = ymax / h
                xmin, ymin, xmax, ymax = self.transform.apply_to_bbox((xmin, ymin, xmax, ymax), **config)
                xmin = int(xmin * w)
                ymin = int(ymin * h)
                xmax = int(xmax * w)
                ymax = int(ymax * h)
                cv2.rectangle(imgs[i], (xmin, ymin), (xmax, ymax), (0, 0, 255))
            cv2.imshow("motion affine", imgs[i])
            key = cv2.waitKey(0)
            if key == ord("q"):
                exit()
            cv2.destroyAllWindows()

    def random_easing_fn(self):
        if random.random() < 0.2:
            return ef.LinearInOut()
        else:
            return random.choice([
                ef.BackEaseIn,
                ef.BackEaseOut,
                ef.BackEaseInOut,
                ef.BounceEaseIn,
                ef.BounceEaseOut,
                ef.BounceEaseInOut,
                ef.CircularEaseIn,
                ef.CircularEaseOut,
                ef.CircularEaseInOut,
                ef.CubicEaseIn,
                ef.CubicEaseOut,
                ef.CubicEaseInOut,
                ef.ExponentialEaseIn,
                ef.ExponentialEaseOut,
                ef.ExponentialEaseInOut,
                ef.ElasticEaseIn,
                ef.ElasticEaseOut,
                ef.ElasticEaseInOut,
                ef.QuadEaseIn,
                ef.QuadEaseOut,
                ef.QuadEaseInOut,
                ef.QuarticEaseIn,
                ef.QuarticEaseOut,
                ef.QuarticEaseInOut,
                ef.QuinticEaseIn,
                ef.QuinticEaseOut,
                ef.QuinticEaseInOut,
                ef.SineEaseIn,
                ef.SineEaseOut,
                ef.SineEaseInOut,
                Step,
            ])()

class Step: # Custom easing function for sudden change.
    def __call__(self, value):
        return 0 if value < 0.5 else 1


if __name__ == "__main__":
    import time
    random_brightness = RandomContrast()
    # random_brightness_v2 = RandomContrastv2()

    img = cv2.imread("./test.jpg")
    img_copy = copy.deepcopy(img)
    t1 = time.time()
    img = random_brightness(img)
    print("random_brightness_v1: ", (time.time() - t1) * 1000)
    t1 = time.time()
    img_copy = random_brightness_v2(img_copy)
    print("random_brightness_v2: ", (time.time() - t1) * 1000)
    cv2.imshow("random_brightness_v1", img)
    cv2.imshow("random_brightness_v2", img_copy)
    print((img == img_copy).all())
    key = cv2.waitKey(0)
