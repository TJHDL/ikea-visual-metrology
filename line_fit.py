# -*- coding: utf-8 -*-
# @Time    : 2023/8/9 22:06
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : line_fit.py

import cv2
import numpy as np
import os
from semantic_info import LEDNet_inference
import random

img_dir = r'C:\Users\95725\Desktop\src\kuwei18'
img_name = '3.jpg'
save_dir = r'F:\ProjectImagesDataset\IKEA\20230729\rtsp_0729_middle\line_fit_result'


def fit_line_test(img_dir, image_name):
    # 读取图片
    image = cv2.imread(os.path.join(img_dir, image_name))
    # image = cv2.resize(image, (512, 512))

    # 将图片转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # 进行霍夫直线变换
    # default threshold: threshold=50, minLineLength=50, maxLineGap=5
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=150, minLineLength=120, maxLineGap=5)

    # 进行语义分割
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predict, mask = LEDNet_inference(RGB_image)

    # 在原图上绘制检测到的直线
    min_b = 512
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        k = abs((y2 - y1) / (x2 - x1))
        b = y1 - k * x1
        semantic_prob = semantic_filter(predict, k, b, x1, x2)
        if k > 0.2 or semantic_prob <= 0.5:
            continue
        if b < min_b:
            min_b = b
            tar_x1, tar_y1, tar_x2, tar_y2 = x1, y1, x2, y2

    cv2.line(image, (tar_x1, tar_y1), (tar_x2, tar_y2), (0, 255, 0), 2)
    tar_k = abs((tar_y2 - tar_y1) / (tar_x2 - tar_x1))
    tar_b = tar_y1 - k * tar_x1

    # # 显示结果
    cv2.imshow('Detected Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite(os.path.join(save_dir, image_name), image)


def fit_line(image, predict):
    # 将图片转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # 进行霍夫直线变换
    # default threshold: threshold=150, minLineLength=80, maxLineGap=5
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=150, minLineLength=90, maxLineGap=5)

    # 在原图上绘制检测到的直线
    min_b = image.shape[0]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        k = abs((y2 - y1) / (x2 - x1))
        b = y1 - k * x1
        semantic_prob = semantic_filter(predict, k, b, x1, x2)
        if k > 0.2 or semantic_prob <= 0.5:
            continue
        if b < min_b:
            min_b = b
            tar_x1, tar_y1, tar_x2, tar_y2 = x1, y1, x2, y2

    image_line = image.copy()
    cv2.line(image_line, (tar_x1, tar_y1), (tar_x2, tar_y2), (0, 255, 0), 2)
    tar_k = (tar_y2 - tar_y1) / (tar_x2 - tar_x1)
    tar_b = tar_y1 - tar_k * tar_x1

    return tar_k, tar_b, image_line

def fit_line_experiment_verision(image, predict, save_dir, image_name):
    # 将图片转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # 进行霍夫直线变换
    # default threshold: threshold=150, minLineLength=80, maxLineGap=5
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=150, minLineLength=90, maxLineGap=5)

    # 在原图上绘制检测到的直线
    image_line = image.copy()
    min_b = image.shape[0]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        k = abs((y2 - y1) / (x2 - x1))
        b = y1 - k * x1
        semantic_prob = semantic_filter(predict, k, b, x1, x2)
        if k > 0.2 or semantic_prob <= 0.5:
            continue
        if b < min_b:
            min_b = b
            tar_x1, tar_y1, tar_x2, tar_y2 = x1, y1, x2, y2

    # cv2.line(image_line, (tar_x1, tar_y1), (tar_x2, tar_y2), (255, 0, 0), 2)
    # cv2.imwrite(os.path.join(save_dir, image_name), image_line)
    try:
        tar_k = (tar_y2 - tar_y1) / (tar_x2 - tar_x1)
        tar_b = tar_y1 - tar_k * tar_x1
    except Exception as e:
        return -1, -1, image_line

    return tar_k, tar_b, image_line


def upside_line_detect(image, predict, p0_x, p0_y, p1_x, p1_y):
    k, b, image_line = fit_line(image, predict)
    p2_x = p0_x
    p2_y = int(k * p2_x + b)
    p3_x = p1_x
    p3_y = int(k * p3_x + b)

    return p2_x, p2_y, p3_x, p3_y, image_line

def upside_line_detect_experiment_version(image, predict, p0_x, p0_y, p1_x, p1_y, save_dir, image_name):
    k, b, image_line = fit_line_experiment_verision(image, predict, save_dir, image_name)
    if k == -1 and b == -1:
        return -1, -1, -1, -1, image_line
    p2_x = p0_x
    p2_y = int(k * p2_x + b)
    p3_x = p1_x
    p3_y = int(k * p3_x + b)

    return p2_x, p2_y, p3_x, p3_y, image_line

def semantic_filter(predict, k, b, x1, x2):
    b = b + 3 if b <= 508 else b
    start_x = min(x1, x2)
    end_x = max(x1, x2)

    crossbeam_cnt = 0
    for x in range(start_x, end_x, 1):
        y = min(int(k * x + b), 511)
        if predict[y, x] == 2:
            crossbeam_cnt += 1

    return crossbeam_cnt / (end_x - start_x)


def batch_process():
    for idx, file in enumerate(os.listdir(img_dir)):
        try:
            print("------""Processing file " + str(idx) + ": " + file + "......------")
            fit_line_test(img_dir, file)
        except Exception as e:
            print(repr(e))

if __name__ == '__main__':
    fit_line_test(img_dir, img_name)
    # batch_process()