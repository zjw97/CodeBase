import cv2
import numpy as np
import math

#brenner梯度函数计算
def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        for y in range(0, shape[1]):
            out+=(int(img[x+2,y])-int(img[x,y]))**2
    return out

#Laplacian梯度函数计算
def Laplacian(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    return cv2.Laplacian(img,cv2.CV_64F).var()

#SMD梯度函数计算
def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0]-1):
        for y in range(0, shape[1]):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))
    return out

#SMD2梯度函数计算
def SMD2(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    return out

#方差函数计算
def variance(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            out+=(img[x,y]-u)**2
    return out

#energy函数计算
def energy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)*((int(img[x,y+1]-int(img[x,y])))**2)
    return out

#Vollath函数计算
def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(img[x,y])*int(img[x+1,y])
    return out

#entropy函数计算
def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out

def main(img1):
    print('Brenner',brenner(img1))
    print('Laplacian',Laplacian(img1))
    print('SMD',SMD(img1))
    print('SMD2',SMD2(img1))
    print('Variance',variance(img1))
    print('Energy',energy(img1))
    print('Vollath',Vollath(img1))
    print('Entropy',entropy(img1))
import os
import shutil
if __name__ == '__main__':
    path = '/data2/competion/data/train/image'
    #xml_path = '/data2/competion/data/train/box'
    write_path_clear = '/data2/competion/data/train/image_clear'
    write_path_blur = '/data2/competion/data/train/image_blur'
    for file_name in os.listdir(path):
        #print('\n')
        
        img_path = os.path.join(path, file_name)
        img_clear_path = os.path.join(write_path_clear, file_name)
        img_blurr_path = os.path.join(write_path_blur, file_name)
        img_frame = cv2.imread(img_path)
        img_frame =cv2.resize(img_frame,(960,960))
        img_frame = cv2.cvtColor(img_frame,cv2.COLOR_BGR2GRAY)
        clean_du = brenner(img_frame)
        if clean_du> 20000088:
            print('qingxi1 --------------',file_name)
            shutil.copy(img_path, img_clear_path)
        else:
            print('mohu --------------',file_name)
            shutil.copy(img_path, img_blurr_path)
        
        #main(img_frame)
