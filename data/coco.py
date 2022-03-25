import cv2
import cv2
import os
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

if __name__ == "__main__":
    img_path = '/home/zjw/Datasets/coco2017/train2017'
    annFile = '/home/zjw/Datasets/coco2017/annotations/instances_train2017.json'

    coco = COCO(annFile)
    imshow(coco, img_path)




