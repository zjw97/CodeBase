import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

"""
    coco bbox prediction format:
    [{
        "image_id": int, 
        "category_id": int, 
        "bbox": [x,y,width,height], # 对应的iouType关键字"bbox"
        "segmentation": RLE,        # segm
        "keypoints": [x1,y1,v1,...,xk,yk,vk],  # keypoints
        "score": float,
    }]
    
"""

def get_bbox_map_coco(annotation_path, prediction_path):
    coco = COCO(annotation_path)
    coco_pred = coco.loadRes(prediction_path)
    coco_evaluator = COCOeval(coco, coco_pred, iouType="bbox")
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()