# import os
# import cv2
# import random
# import numpy as np
# from remo.data import parse_remo_xml
#
# class RandomPaste():
#
#     def __init__(self):
#         self.image_list = self.load_image_list()
#         self.data_path = "/home/zjw/Datasets/Lvis_other_dataset_withoutPersonPic"
#         self.scale = (0.6, 0.8)
#
#     def load_image_list(self):
#         image_list = []
#         with open("/home/zjw/Datasets/Lvis_other_dataset_withoutPersonPic/Lvis_other_dataset_withoutPersonPic.txt") as f:
#             for line in f:
#                 image_id = os.path.basename(line.split(" ")[0]).split(".")[0]
#                 image_list.append(image_id)
#         return image_list
#
#     def __call__(self, image, boxes):
#         num_box = len(boxes)
#         boxes_copy = boxes.astype("int")
#         # 随机选择一个box
#         box_idx = random.randint(0, num_box-1)
#         box = boxes_copy[box_idx, :]
#         xmin, ymin, xmax, ymax, cid = box[:]
#         box_width = xmax - xmin
#         box_height = ymax - ymin
#
#         scale = random.uniform(self.scale[0], self.scale[1])
#         mask_id = random.choice(self.image_list)
#         mask_path = os.path.join(self.data_path, "masks/" + mask_id + ".png")
#         mask = cv2.imread(mask_path)
#         texture_path = os.path.join(self.data_path, "images/" + mask_id + ".jpg")
#         texture = cv2.imread(texture_path)
#         mask = cv2.resize(mask, (int(box_width * scale), int(box_height * scale)))
#         texture = cv2.resize(texture, (int(box_width * scale), int(box_height * scale)))
#
#         texture *= mask
#
#         paste_x = xmin + int(box_width * random.uniform(0, 1 - scale))
#         paste_y = ymin + int(box_height * random.uniform(0, 1 - scale))
#
#         scenic_mask = (~(mask * 255) / 255).astype("uint8")
#         image[paste_y:paste_y + int(box_height * scale), paste_x:paste_x + int(box_width * scale)] *= scenic_mask
#         image[paste_y:paste_y + int(box_height * scale), paste_x:paste_x + int(box_width * scale)] += texture
#         return image
#
# def resize_image(image, boxes):
#     h, w, c = image.shape
#     image = cv2.resize(image, (512, 288))
#     boxes[:, ::2] = boxes[:, ::2] / w * 512
#     boxes[:, 1::2] = boxes[:, 1::2] / h * 288
#     return image, boxes
#
#
# if __name__ == "__main__":
#     random.seed(2022)
#     np.random.seed(2022)
#     xml_list = "/home/zjw/Datasets/AIC_REMOCapture/txt/trainval_AIC_remocap2018053008070827.txt"
#     data_path = "/home/zjw/Datasets/AIC_REMOCapture"
#
#     random_paste = RandomPaste()
#
#     image_list = []
#     with open(xml_list, "r") as f:
#         for line in f:
#             image_list.append(os.path.join(data_path, line.strip()))
#
#     idx = 0
#     while True:
#         xml_path = image_list[idx]
#         meta = parse_remo_xml(data_path, xml_path)
#         boxes = np.array(meta["boxes"], dtype="float")
#         image = cv2.imread(meta["image_path"])
#         image, boxes = resize_image(image, boxes)
#         image = random_paste(image, boxes)
#         cv2.imshow("merged", image)
#         key = cv2.waitKey(0)
#         if key == ord('q'):
#             exit()
#         elif key == ord("a"):
#             idx -= 1
#         else:
#             idx += 1


import time
import numpy as np
import random

x = list(range(10))
t1 = time.time()
random.choice(x)
print((time.time() - t1) * 10000)