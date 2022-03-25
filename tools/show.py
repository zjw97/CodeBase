# -*- coding: UTF-8 -*-
import numpy as np
from PIL import Image
import cv2
import numpy as np
import math
import seaborn as sns

__all__ = ["draw_rectangle", "put_text", "putImgToOne", "draw_boxes", "color_palette"]

def read_img(img_file, dtype=np.float32, color=True):

    f = Image.open(img_file) # 返回的是一个fp类型
    try:
        if color:
            img = f.convert("RGB")
        else :
            img = f.convert("P")
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, "close"):
            f.close()

    if img.ndim == 2:
        return img[np.newaxis]
    else: # transpose H, W, C -> C,H, W
        return img.transpose(2, 0, 1)

def color_palette(palette="hls", n_colors=10):
    current_pale = sns.color_palette(palette, n_colors)
    colors = []
    for i in range(n_colors):
        colors.append([int(i * 255) for i in current_pale[i]])
    return colors

def draw_rectangle(image, org1, org2, color, thickness=1):
    cv2.rectangle(img=image, pt1=org1, pt2=org2, color=color, thickness=thickness, lineType=cv2.LINE_4)

def put_text(img, text, org, color=(0, 0, 255), thickness=1):
    cv2.putText(img, text, org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.,
                color=color, thickness=thickness, lineType=cv2.LINE_8)

def draw_boxes(image, boxes, conf=None, label=None):
    palette = color_palette()
    num_box = boxes.shape[0]
    if boxes.shape[1]== 5:
        for i in range(num_box):
            xmin, ymin, xmax, ymax, cid = boxes[i, :]
            if cid not in [1, 3]:
                continue
            draw_rectangle(image, (xmin, ymin), (xmax, ymax), color=palette[cid], thickness=1)
            put_text(image, "%d"%(cid), (xmin, ymin), color=palette[cid], thickness=1)
    else:
        for i in range(num_box):
            xmin, ymin, xmax, ymax = boxes[i, :]
            draw_rectangle(image, (xmin, ymin), (xmax, ymax), color=palette[label[i]], thickness=1)
            put_text(image, "%.4f"%(conf[i]), (xmin, ymin), color=palette[label[i]], thickness=1)

def putImgToOne(all_images: list, n_cols=2):
    # 设置显示窗口的最大大小
    w_max_window = 1280
    h_max_window = 720

    img_max_width = 0
    img_max_height = 0
    for img in all_images:
        height, width, _ = img.shape
        if width > img_max_width:
            img_max_width = width
        if height > img_max_height:
            img_max_height = height

    num_imgs = len(all_images)
    if num_imgs < n_cols:
        n_cols = num_imgs  # 列数，每行几张图片
        n_rows = 1  # 行数， 每列几张图片
    else:
        n_rows = int(math.ceil(num_imgs / n_cols))

    x_space = 5
    y_space = 5

    oneImage = np.zeros((n_rows * img_max_height + y_space * (n_rows - 1),
                         n_cols * img_max_width + x_space * (n_cols - 1), 3)).astype(np.uint8)

    cnt_imgs = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if cnt_imgs >= num_imgs:
                break

            img = all_images[cnt_imgs]
            height, width, channel = img.shape
            if channel == 1:
                img_new = np.zeros((height, width, 3)).astype(np.uint8)
                img_new[:, :, 0] = img
                img_new[:, :, 1] = img
                img_new[:, :, 2] = img
                img = img_new

            xmin = j * (img_max_width + x_space)
            ymin = i * (img_max_height + y_space)
            xmax = xmin + width
            ymax = ymin + height

            oneImage[ymin: ymax, xmin: xmax, :] = img
            cnt_imgs += 1
    scale_x = w_max_window / (img_max_width * n_cols)
    scale_y = h_max_window / (img_max_height * n_rows)
    scale = min(scale_x, scale_y)
    oneImage = cv2.resize(oneImage, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return oneImage


# def nothing(*args):
#     pass
#
# def add_track_bar(video_name, pos):
#     cap = cv2.VideoCapture(video_name)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#     n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     cv2.createTrackbar("time", "result", pos, n_frames, nothing)
#
#     while True:
#         t_pos = cv2.getTrackbarPos("time", "result")
#         if t_pos + 1 == pos:  # 如果trackbar的位置没有被拖动, 那么上一次读取图片之后pos指向下一个位置, 此时track bar的位置还为更新
#             cv2.setTrackbarPos("time", "result", pos)
#         else:  # 如果进度条被拖动了, 那么需要重新设置读取图片的点, 进度条的位置不需要修改
#             pos = t_pos
#             cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
#         # 之后就是正常读图片遍历的过程了
#         _, image = cap.read()
#
#
# # 在opencv窗口画图
# def draw_circle(event, x, y, flags, param):
#     # 获取四个滑动条的位置
#     r = cv2.getTrackbarPos('R', 'image')
#     g = cv2.getTrackbarPos('G', 'image')
#     b = cv2.getTrackbarPos('B', 'image')
#     color = (b, g, r)
#
#     global position, drawing, mode
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         position = (x, y)  # 当按下左键是返回起始位置坐标
#
#     elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:  # 鼠标左键拖拽
#         if drawing == True:
#             if mode == True:
#                 cv2.rectangle(img, position, (x, y), color, -1)  # 绘制矩形
#             else:
#                 cv2.circle(img, (x, y), 2, color, -1)  # 绘制圆圈，小圆点连在一起就成了线，3代表了笔画的粗细
#
#     elif event == cv2.EVENT_LBUTTONUP:  # 鼠标松开停止绘画
#         drawing = False
#
#
# # 当按下鼠标时变为True
# drawing = False
# # 如果mode为true时候绘制矩形，按下m变成绘制曲线
# mode = True
#
# img = cv2.imread(r'C:\Users\x\Desktop\12.jpg', cv2.IMREAD_ANYCOLOR)
# cv2.namedWindow('image')
#
# # 创建改变颜色的滑动条
# cv2.createTrackbar('R', 'image', 0, 255, nothing)
# cv2.createTrackbar('G', 'image', 0, 255, nothing)
# cv2.createTrackbar('B', 'image', 0, 255, nothing)
# cv2.setMouseCallback('image', draw_circle)
#
# while (1):
#     cv2.imshow('image', img)
#
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('m'):
#         mode = not mode
#     elif k == 27:
#         break
#
# cv2.destroyAllWindows()