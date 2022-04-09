#!/usr/bin/env python
# encoding: utf-8
"""
    导向滤波应用: 暗通道去雾, 图像Matting
"""
import numpy as np

def boxfilter(img, r):
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)

    imCum = np.cumsum(img, 0)
    imDst[0: r+1, :] = imCum[r: 2*r+1, :]
    imDst[r+1: rows-r, :] = imCum[2*r+1: rows, :] - imCum[0: rows-2*r-1, :]
    imDst[rows-r: rows, :] = np.tile(imCum[rows-1, :], [r, 1]) - imCum[rows-2*r-1: rows-r-1, :]

    imCum = np.cumsum(imDst, 1)
    imDst[:, 0: r+1] = imCum[:, r: 2*r+1]
    imDst[:, r+1: cols-r] = imCum[:, 2*r+1: cols] - imCum[:, 0: cols-2*r-1]
    imDst[:, cols-r: cols] = np.tile(imCum[:, cols-1], [r, 1]).T - imCum[:, cols-2*r-1: cols-r-1]

    return imDst

def guidedfilter(I, p, r, eps):
    (rows, cols) = I.shape
    N = boxfilter(np.ones([rows, cols]), r)
    # print(N)
    meanI = boxfilter(I, r) / N
    meanP = boxfilter(p, r) / N
    meanIp = boxfilter(I * p, r) / N
    covIp = meanIp - meanI * meanP

    meanII = boxfilter(I * I, r) / N
    varI = meanII - meanI * meanI

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = boxfilter(a, r) / N
    meanB = boxfilter(b, r) / N

    q = meanA * I + meanB
    return q

if __name__ == "__main__":
    import cv2
    image = cv2.imread("/home/zjw/CodeBase/论文复现/data/images/neural-style/candy.jpg", cv2.IMREAD_GRAYSCALE)
    image = image.astype("float") / 255
    i = image
    # i = np.zeros_like(image)
    i = np.ones_like(image)
    img = guidedfilter(i, image, 2, 1e-8)
    img = (img * 255).astype("uint8")
    cv2.imshow("raw image", image)
    cv2.imshow("guided filter", img)
    key = cv2.waitKey(0)
