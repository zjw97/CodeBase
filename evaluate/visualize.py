# -*- coding: UTF-8 -*-
# path: c:/Users/zjw/Desktop/codeBase/evaluate
import cv2



def draw_det_bboxes(image, bboxes, color):
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, category, score = bbox
        cv2.rectangle(img=image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=1,lineType=cv2.LINE_4)
        cv2.putText(img=image, text=str(score), org=(xmin, ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2,
                    color=color, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
    return image

def draw_gt_bboxes(image, bboxes, color):
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, category = bbox
        cv2.rectangle(img=image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=1,lineType=cv2.LINE_4)
        # cv2.putText(img=image, text=str(category), org=(xmin, ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2,
        #             color=color, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
    return image