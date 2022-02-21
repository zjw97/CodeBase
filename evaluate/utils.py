import cv2
import numpy as np
import math
import seaborn as sns

__all__ = ["color_palette", "draw_rectangle", "put_text", "putImgToOne"]

def color_palette(palette="hls", n_colors=10):
    current_pale = sns.color_palette(palette, n_colors)
    colors = []
    for i in range(n_colors):
        colors.append([int(i * 255) for i in current_pale[i]])
    return colors

def draw_rectangle(image, org1, org2, color, thickness=1):
    cv2.rectangle(img=image, pt1=org1, pt2=org2, color=color, thickness=thickness, lineType=cv2.LINE_4)
    return image

def put_text(img, text, org, color=(0, 0, 255), thickness=1):
    cv2.putText(img, text, org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.,
                color=color, thickness=thickness, lineType=cv2.LINE_8)


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