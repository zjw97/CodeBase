# -*- coding: UTF-8 -*-
# windows path: c:/Users/zjw/Desktop/codeBase/evaluate
# linux path: /home/zjw/zjw/CodePath
import copy
import os
import cv2
import json

from tools import *

def parse_coco_annotations(annotations):
    images = annotations["images"]
    anno = annotations["annotations"]
    id2image = {}
    for image in images:
        image_id = image["id"]
        file_name = image["file_name"]
        width = image["width"]
        height = image["height"]
        id2image[image_id] = {
            "file_name": file_name,
            "width": width,
            "height": height,
            "bbox": [],
        }
    for box_anno in anno:
        image_id = box_anno["image_id"]
        gt_bbox = box_anno["bbox"]
        cid = box_anno["category_id"]
        gt_bbox = gt_bbox + [cid]
        id2image[image_id]["bbox"].append(gt_bbox)

    return id2image

def parse_coco_inference(predictions):
    id2pred = {}
    for pred in predictions:
        image_id = pred["image_id"]
        cid = pred["category_id"]
        bbox = pred["bbox"]
        score = pred["score"]
        if image_id not in id2pred.keys():
            id2pred[image_id] = [bbox + [cid, score]]
        else:
            id2pred[image_id].append(bbox + [cid, score])

    return id2pred


def visualize_coco_inference(image_root, anno_file_, prediction_file):
    annotations = json.load(open(anno_file, "r"))
    annotations = parse_coco_annotations(annotations)
    predictions = json.load(open(prediction_file, "r"))
    predictions = parse_coco_inference(predictions)



if __name__ == "__main__":
    def preprocess(image, scale = 2):
        height, width, _ = image.shape
        expand_width = int(max(width, height / 9 * 16)) * scale
        expand_height = int(max(height, width / 16 * 9)) * scale
        expand_bottom = expand_height - height
        expand_right = expand_width - width
        image = cv2.copyMakeBorder(image, 0, expand_bottom, 0, expand_right, cv2.BORDER_CONSTANT)
        image = cv2.resize(image, (512, 288), 0, 0, cv2.INTER_NEAREST)
        return image


    image_root = "/home/zjw/Datasets/Multiview_Hand_Fusion_Dataset/images/val" 
    anno_file = "/home/zjw/REMO/PytorchSSD/multiview_handdet_anno_coco_scale_2x.json"
    annotations = json.load(open(anno_file, "r"))
    annotations = parse_coco_annotations(annotations)

    prediction_list = []
    prediction_file_list = []
    show_string = []
    # prediction_file_list.append("/home/zjw/REMO/0022_PytorchHanddet/hand_det_norm_flip_lrmult_conf_branch_divby_num_conf/models/modeliter_150000_coco_scale_2x.json")
    # show_string.append("hand_det_conf_weight_150k")

    prediction_file_list.append(
        "/home/zjw/REMO/0022_PytorchHanddet/hand_det_norm_flip_lrmult_revise_anchor/models/modeliter_50000_coco_scale_2x.json")
    show_string.append("hand_det_revise_anchor_50k")

    for prediction_file in prediction_file_list:
        prediction_list.append(parse_coco_inference(json.load(open(prediction_file, "r"))))

    for image_id, anno in annotations.items():
        file_name = anno["file_name"]
        image_path = os.path.join(image_root, file_name)
        image = cv2.imread(image_path)
        image = preprocess(image)

        width = anno["file_name"]
        height = anno["height"]
        gt_bboxes = anno["bbox"]
        show_image_list = []
        for box in gt_bboxes:
            xmin, ymin, width, height, cid = box[:]
            xmax = int(xmin + width)
            ymax = int(ymin + height)
            print("file_name: %s, gt box width: %d height: %d" % (file_name, int(width), int(height)))
            draw_rectangle(image, (int(xmin), int(ymin)), (xmax, ymax), (0, 0, 255), thickness=2)

        for pred_idx, prediction in enumerate(prediction_list):
            show_image = copy.deepcopy(image)
            try:
                pred_boxes = prediction[image_id]
            except:
                pred_boxes = []
            put_text(show_image, show_string[pred_idx], (5, 20), (255, 255, 0))
            for pred_box in pred_boxes:
                xmin, ymin, width, height, cid, score = pred_box[:]
                xmax = int(xmin + width)
                ymax = int(ymin + height)
                draw_rectangle(show_image, (int(xmin), int(ymin)), (xmax, ymax), (255, 255, 0))
            show_image_list.append(show_image)

        all_image = putImgToOne(show_image_list)
        cv2.imshow(file_name, all_image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        cv2.destroyAllWindows()



