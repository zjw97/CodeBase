import os.path
import random
from xml.dom.minidom import Document
import xml.etree.ElementTree as ET

import cv2

from evaluate import draw_rectangle, put_text

def write_xml(anno, xml_name):
    # Create the minidom document
    doc = Document()

    # Create the root element: annotation
    anno_root = doc.createElement("Annotations")
    doc.appendChild(anno_root)

    # ImagePath
    image_path_node = doc.createElement("ImagePath")
    image_path = doc.createTextNode(anno['image_path'])
    image_path_node.appendChild(image_path)
    anno_root.appendChild(image_path_node)

    # Dataset
    dataset_node = doc.createElement("DataSet")
    dataset = doc.createTextNode(anno['data_set'])
    dataset_node.appendChild(dataset)
    anno_root.appendChild(dataset_node)

    # ImageWidth
    width_node = doc.createElement("ImageWidth")
    width = doc.createTextNode("%d" % anno["width"])
    width_node.appendChild(width)
    anno_root.appendChild(width_node)
    # ImageHeight
    height_node = doc.createElement("ImageHeight")
    height = doc.createTextNode("%d" % anno["height"])
    height_node.appendChild(height)
    anno_root.appendChild(height_node)

    # Num of parts
    np_node = doc.createElement("NumPerson")
    np = doc.createTextNode("%d" % anno["num"])
    np_node.appendChild(np)
    anno_root.appendChild(np_node)
    # Objects
    for i in range(anno["num"]):
        pNode = doc.createElement("Object_{}".format(i+1))
        # cid
        cid_node = doc.createElement("cid")
        cid = doc.createTextNode("%d" % anno["boxes"][i]["cid"])
        cid_node.appendChild(cid)
        pNode.appendChild(cid_node)
        # xmin
        xmin_node = doc.createElement("xmin")
        xmin = doc.createTextNode("%.2f" % anno["boxes"][i]["xmin"])
        xmin_node.appendChild(xmin)
        pNode.appendChild(xmin_node)
        # ymin
        ymin_node = doc.createElement("ymin")
        ymin = doc.createTextNode("%.2f" % anno["boxes"][i]["ymin"])
        ymin_node.appendChild(ymin)
        pNode.appendChild(ymin_node)
        # xmax
        xmax_node = doc.createElement("xmax")
        xmax = doc.createTextNode("%.2f" % anno["boxes"][i]["xmax"])
        xmax_node.appendChild(xmax)
        pNode.appendChild(xmax_node)
        # ymax
        ymax_node = doc.createElement("ymax")
        ymax = doc.createTextNode("%.2f" % anno["boxes"][i]["ymax"])
        ymax_node.appendChild(ymax)
        pNode.appendChild(ymax_node)
        # add this object
        anno_root.appendChild(pNode)
    with open(xml_name, "wb") as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))

def parse_remo_xml(data_root, xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    meta = {}
    meta["image_path"] = os.path.join(data_root, root.find("ImagePath").text)
    meta["width"] = int(root.find("ImageWidth").text)
    meta["height"] = int(root.find("ImageHeight").text)
    meta["num_person"] = int(root.find("NumPerson").text)
    meta["dataset"] = root.find("DataSet").text
    meta["boxes"] = []
    for i in range(meta["num_person"]):
        obj = root.find("Object_%d"%(i+1))
        xmin = float(obj.find("xmin").text)
        ymin = float(obj.find("ymin").text)
        xmax = float(obj.find("xmax").text)
        ymax = float(obj.find("ymax").text)
        cid = int(obj.find("cid").text)
        meta["boxes"].append([xmin, ymin, xmax, ymax, cid])
    return meta

def remo_visualize(data_root, xml_path_list, shuffle=False):
    xml_list = []
    with open(os.path.join(data_root, xml_path_list)) as f:
        for line in f:
            line = line.strip()
            xml_list.append(os.path.join(data_root, line))
    if shuffle:
        random.shuffle(xml_list)

    line_idx = 0
    while True:
        xml_path = xml_list[line_idx]
        meta = parse_remo_xml(data_root, xml_path)
        img_path = meta["image_path"]
        img = cv2.imread(img_path)
        width = meta["width"]
        height = meta["height"]
        boxes = meta["boxes"]
        for box in boxes:
            xmin, ymin, xmax, ymax, cid = box[:]
            xmin = int(xmin * width)
            ymin = int(ymin * height)
            xmax = int(xmax * width)
            ymax = int(ymax * height)
            img = draw_rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255))
            put_text(img, str(cid), (xmin, ymax), (0, 0, 255))
        cv2.imshow(img_path, img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        elif key == ord("a"):
            line_idx -= 1
        else:
            line_idx += 1
        cv2.destroyAllWindows()





