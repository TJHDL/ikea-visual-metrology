# -*- coding: utf-8 -*-
# @Time    : 2024/1/24 18:57
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : guided_filter.py
import cv2
import os
import numpy as np

image_dir = r'C:\Users\95725\Desktop\guided_filter\image'
depth_dir = r'C:\Users\95725\Desktop\guided_filter\depth'
save_dir = r'C:\Users\95725\Desktop\guided_filter\result'
sample_num = '1103'

def guided_filter(I, p, radius, eps):
    """
    Guided filter implementation.

    Parameters:
        - I: Guiding image (single channel).
        - p: Filtering input image.
        - radius: Radius of the filter.
        - eps: Regularization parameter.

    Returns:
        - q: Filtered output.
    """
    # Compute the mean of I, p, and I*p
    mean_I = cv2.boxFilter(I, -1, (radius, radius))
    mean_p = cv2.boxFilter(p, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(I * p, -1, (radius, radius))

    # Compute the covariance of I and p
    cov_Ip = mean_Ip - mean_I * mean_p

    # Compute the mean of I^2 and the variance of I
    mean_II = cv2.boxFilter(I * I, -1, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    # Compute the a and b parameters
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Compute the mean of a and b
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    # Compute the filtered output q
    q = mean_a * I + mean_b

    return q

def fill_depth_holes(depth_map, max_hole_size=100):
    """
    使用基于图像的方法填充深度图中的孔洞。

    Parameters:
        - depth_map: 输入深度图。
        - max_hole_size: 允许的最大孔洞尺寸。

    Returns:
        - filled_depth_map: 填充后的深度图。
    """
    # 寻找深度图中的孔洞
    holes = depth_map == 0

    # 标记孔洞，用不同的整数标记不同的孔洞
    _, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8))

    # 寻找最大的孔洞
    max_hole_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # 填充除最大孔洞之外的其他孔洞
    holes_to_fill = labels == max_hole_index
    filled_depth_map = depth_map.copy()
    filled_depth_map[holes_to_fill] = cv2.inpaint(depth_map, holes_to_fill.astype(np.uint8), max_hole_size, cv2.INPAINT_TELEA)

    return filled_depth_map

if __name__ == '__main__':
    # 读取深度图
    image = cv2.imread(os.path.join(image_dir, sample_num + ".jpg"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.imread(os.path.join(depth_dir, sample_num + ".png"), cv2.IMREAD_GRAYSCALE)

    height = image.shape[0]
    width = image.shape[1]
    w = int(width / 2)
    h = int(height / 2)

    # 将深度图归一化到[0, 1]范围
    normalized_depth = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    normalized_gray = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX)

    # 创建高斯模糊器
    blurred_depth = cv2.GaussianBlur(depth_map, (17, 17), 0)

    # 使用导向滤波平滑深度图
    # smoothed_depth = guided_filter(depth_map, depth_map, radius=5, eps=1e-4)
    # smoothed_depth = smoothed_depth.astype(np.uint8)
    # smoothed_depth[smoothed_depth < 0] = 0

    # 显示原始深度图和滤波后的深度图
    cv2.namedWindow('Original Depth', 0)
    cv2.resizeWindow('Original Depth', w, h)
    cv2.imshow('Original Depth', depth_map)
    cv2.namedWindow('Smoothed Depth', 0)
    cv2.resizeWindow('Smoothed Depth', w, h)
    cv2.imshow('Smoothed Depth', blurred_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
