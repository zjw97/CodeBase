import cv2
import numpy as np

def center_crop_remove_outbound_box(image_width, image_height, boxes):
    crop_width = int(min(image_width, image_height / 9 * 16))
    crop_height = int(min(image_height, image_width / 16 * 9))

    crop_xmin = image_width / 2 - crop_width / 2
    crop_ymin = image_height / 2 - crop_height / 2
    crop_xmax = image_width / 2 + crop_width / 2
    crop_ymax = image_height/ 2 + crop_height / 2

    keep_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax, cid = box
        if xmin < crop_xmin or ymin < crop_ymin or xmax > crop_xmax or ymax > crop_ymax:
            continue
        xmin = max(xmin, crop_xmin)
        ymin = max(ymin, crop_ymin)
        xmax = min(xmax, crop_xmax)
        ymax = min(ymax, crop_ymax)
        keep_boxes.append([xmin, ymin, xmax, ymax, cid])
    return keep_boxes