import cv2
import numpy as np

def plot_feature_map(show_name, num_cols, feature_map):
    _, channels, height, width = feature_map.shape
    feature_map = feature_map.data.cpu().numpy().squeeze()
    num_rows = int(math.ceil(channels / num_cols) )
    img_show = np.zeros((height * num_rows, width * num_cols)).astype(np.float)
    for idx in range(channels):
        ir = idx % num_cols
        ic = idx // num_rows
        xmin = ic * width
        xmax = (ic + 1) * width
        ymin = ir * height
        ymax = (ir + 1) * height
        # 先对通道求绝对值, 这样负响应的可能也显示会比较亮
        # r4npy[idx] = np.abs(r4npy[idx])
        # r4npy[idx] /= r4npy[idx].max()

        # 每个通道自己做归一化来显示
        f_max, f_min = np.max(feature_map, axis=(1, 2), keepdims=True), np.min(feature_map, axis=(1, 2), keepdims=True)
        feature_map[idx] -= f_min[idx]
        feature_map[idx] /= (f_max[idx] - f_min[idx])
        img_show[ymin:ymax, xmin:xmax] = feature_map[idx]
    cv2.namedWindow(show_name, cv2.NORM_HAMMING)
    cv2.imshow(show_name, img_show)
