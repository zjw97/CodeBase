# -*- coding: UTF-8 -*-
import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

__all__ = ["parse_voc_xml"]

def parse_voc_xml(xml):
    tree = ET.parse(xml)
    root = tree.getroot()
    meta = {}
    annotations = root.find("annotations")
    filename = annotations.find("filename").text
    meta["filename"] = filename

    size = annotations.find("size")
    width = int(size.find("width").text)
    meta["width"] = width
    height = int(size.find("height").text)
    meta["height"] = height
    objects = annotations.findall("object")
    meta["num"] = len(objects)
    meta["boxes"] = []

    for obj in objects:
        xmin = int(obj.find("xmin").text)
        ymin = int(obj.find("ymin").text)
        xmax = int(obj.find("xmax").text)
        ymax = int(obj.find("ymax").text)
        meta["boxes"].append({
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "cid": 1
        })
    return meta