import os
import numpy as np
import pandas as pd
import seaborn as sns

from remo.data import parse_remo_xml

def generator_csv():
    xml_root_list = [
        "/home/zhangming/Datasets/AIC_REMOCapture",
        "/home/zhangming/Datasets/RemoCoco",
        "/home/zhangming/Datasets/OtherBackGround_Images",
        "/home/zhangming/Datasets/EgoHand",
        "/home/zhangming/Datasets/Multiview_Hand_Pose_Dataset",
        "/home/zhangming/Datasets/Multiview_Hand_Fusion_Dataset",
    ]
    xml_list_file = [
        "/home/zhangming/Datasets/AIC_REMOCapture/txt/trainval_AIC_remocap2018053008070827.txt"
        "/home/zhangming/Datasets/RemoCoco/Layout/trainval_NoPersonClean.txt"
        "/home/zhangming/Datasets/OtherBackGround_Images/background_list.txt"
        "/home/zhangming/Datasets/EgoHand/egohand_data.txt"
        "/home/zhangming/Datasets/Multiview_Hand_Pose_Dataset/Multiview_Hand_Data.txt"
        "/home/zhangming/Datasets/Multiview_Hand_Fusion_Dataset/train_hand.txt"
    ]

    xml_file_list = []
    for file_idx, xml_file in enumerate(xml_list_file):
        with open(xml_file) as f:
            for line in f:
                line = line.strip()
                xml_file_list.append((xml_root_list[file_idx], line))

    df = pd.DataFrame(columns=["image", "xml", "dataset", "image_width", "image_height",
                               "xmin", "ymin", "xmax", "ymax", "bbox_width", "bbox_height"])
    for root, xml_file_path in xml_file_list:
        dataset = os.path.basename(root)
        xml = os.path.basename(root, xml_file_path)
        xml_file_path = os.path.join(root, xml_file_path)
        meta = parse_remo_xml(root, xml_file_path)
        image_path = meta["image_path"]
        image = os.path.basename(image_path)
        image_width = meta["width"]
        image_height = meta["height"]
        boxes = meta["boxes"]
        for box in boxes:
            xmin, ymin, xmax, ymax, cid = box[:]
            bbox_width = xmax - xmin
            bbox_height = xmax - ymin
            df.append(pd.DataFrame({
                "image": image,
                "xml": xml,
                "dataset": dataset,
                "image_width": image_width,
                "image_height": image_height,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
            }), ignore_index=False)
    df.to_csv("handdet_statistic.csv")


if __name__ == "__main__":
    pass