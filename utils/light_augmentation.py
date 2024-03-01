# -*- coding: utf-8 -*-
# @Time    : 2023/9/2 11:21
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : light_augmentation.py
import os
import cv2
import numpy as np

src_dir = r'C:\Users\95725\Desktop\rtsp_picture_1227_407'
dst_dir = r''
image_name = r'451.jpg'

# 全局直方图均衡化
def hisEqulColor1(img):
    image_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    img = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    return img


# 自适应直方图均衡化
def hisEqulColor2(img):
    image_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(image_yuv[:, :, 0])
    img = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    return img


# 直方图均衡化
def histogramEqualize(src_dir):
    for file in os.listdir(src_dir):
        img = cv2.imread(os.path.join(src_dir, file))
        img1 = hisEqulColor1(img.copy())
        img2 = hisEqulColor2(img.copy())
        res = np.hstack((img, img1, img2))
        cv2.namedWindow("hist_equal_img", 0)
        cv2.imshow("hist_equal_img", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 平衡图片亮度
def lightGrayEqualize(src_dir):
    for file in os.listdir(src_dir):
        bgr = cv2.imread(os.path.join(src_dir, file))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        _, _, hsv_v = cv2.split(hsv)
        _, hls_l, _ = cv2.split(hls)
        lab_l, _, _ = cv2.split(lab)

        hsv_v_mean = hsv_v.mean()
        hls_l_mean = hls_l.mean()
        lab_l_mean = lab_l.mean()

        hsv[:, :, 2] = hsv_v_mean
        hls[:, :, 1] = hls_l_mean
        lab[:, :, 0] = lab_l_mean

        img1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img2 = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
        img3 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        res = np.hstack((bgr, img1, img2, img3))

        cv2.namedWindow("result", 0)
        cv2.imshow("result", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def histeq(image):
    # 计算像素累积分布函数
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    # 将CDF映射回原图像中
    image_eq = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    return image_eq.reshape(image.shape)

def light_enhancement(src_dir, image_name):
    img = cv2.imread(os.path.join(src_dir, image_name))
    img_bright = cv2.convertScaleAbs(img, alpha=3, beta=0)
    cv2.namedWindow("result", 0)
    cv2.imshow("result", img_bright)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # histogramEqualize(src_dir)
    # lightGrayEqualize(src_dir)
    light_enhancement(src_dir, image_name)