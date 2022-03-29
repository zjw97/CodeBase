import json

import cv2
import os
import time
from tqdm import tqdm
from pycocotools.coco import COCO

from tools.show import *

def imshow(coco, img_path):
    imgIds = coco.getImgIds()
    idx = 0
    while True:
        imgId = imgIds[idx]
        image_name = coco.loadImgs(imgId)[0]['file_name']
        img = cv2.imread(os.path.join(img_path, image_name))

        annIds = coco.getAnnIds(imgIds=imgId, catIds=[], iscrowd=None)
        anns = coco.loadAnns(annIds)
        for j in range(len(anns)):
            coordinate = anns[j]['bbox']
            xmin, ymin, width, height = coordinate[:]
            cid = anns[j]['category_id']
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmin + width), int(ymin + height)), (0, 0, 255))
            cv2.putText(img, "%d" % (cid), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, (0, 0, 255), 1)
        cv2.imshow("coco", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif key == ord("a"):
            idx -= 1
        else:
            idx += 1

def _create_info(version, description):
    info = {
        "year": time.strftime("%Y", time.localtime()),
        "version": version,
        "description": description,
        "contributor": None,
        "url": None,
        "date_created": time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
    }
    return info

def _create_license():
    license = {
        "id": None,
        "name": None,
        "url": None,
    }
    return license

def _create_category(class_label):
    categories = []
    for i, category in enumerate(class_label):
        categories.append({
            "id":i,
            "name": category,
            "supercategory":None
        })
    return categories

def generate_coco_format_anno(version, description, class_label, annotations, save_path):
    anno = {}
    anno["info"] = _create_info(version, description)
    anno["license"] = _create_license()
    anno["categories"] = _create_category(class_label)

    anno["images"] = []
    anno["annotations"] = []
    box_id = 0
    for image_id, meta in tqdm(enumerate(annotations)):
        image = os.path.basename(meta["image"])
        file_path = meta["file_path"]
        width = meta["width"]
        height = meta["height"]
        boxes = meta["boxes"]
        anno["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image,
            "license": None,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": time.strftime("%Y-%m-%s_%M-%H-%S", time.localtime()),
        })

        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax, cid = boxes[i]
            anno["annotations"].append({
                "id": box_id,
                "image_id": image_id,
                "category_id": cid,
                "segmentation": None,
                "area": (xmax - xmin) * (ymax - ymin),
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],  # top-left, width, height
                "iscrowd": 0,
            })
            box_id +=1
    json.dump(anno, open(save_path, "w"))



if __name__ == "__main__":
    img_path = '/home/zjw/Datasets/coco2017/train2017'
    annFile = '/home/zjw/Datasets/coco2017/annotations/instances_train2017.json'

    coco = COCO(annFile)
    imshow(coco, img_path)




