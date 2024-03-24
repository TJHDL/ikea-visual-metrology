# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 16:34
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : vertical_main.py
import os
import cv2
import numpy as np
import config
import torch
import pandas as pd
import matplotlib.pyplot as plt
from model import DirectionalPointDetector
from torchvision.transforms import ToTensor
from data import get_predicted_points, calc_point_squre_dist
from semantic_info import LEDNet_inference, upside_detect, gap_height_measurement_mask, gap_height_measurement_based_on_camera_height, fixed_error_correction
from line_fit import upside_line_detect
from depth.depth_predict import load_model, depth_estimation, plane_to_pointcloud
from utils import get_file_description, close_file_description

image_dir = r'C:\Users\95725\Desktop\src\kuwei18'
save_dir = r'C:\Users\95725\Desktop\dst\kuwei18'
data_src_dir = r'C:\Users\95725\Desktop\src'
data_dst_dir = r'C:\Users\95725\Desktop\dst'
# image_dir = r'D:\ProjectCodes\VisionMeasurement\1020test\floor4\src\kuwei16'
# save_dir = r'D:\ProjectCodes\VisionMeasurement\1020test\floor4\dst\kuwei16'
BoxMPR_detector_weights = r'checkpoints\dp_detector_59_dark.pth'   #开灯:r'checkpoints\dp_detector_799_v100.pth' 关灯:r'checkpoints\dp_detector_59_dark.pth'
LEDNet_detector_weights = r'checkpoints\LEDNet_iter_170400_v100.pth'    #r'checkpoints\LEDNet_iter_170400_v100.pth'
# image_name = r'IMG_0002_box1.jpg'
# point_0_x = 945
# point_0_y = 650
# point_1_x = 2150
# point_1_y = 620

RED_WIDTH = 10 #cm
TIEPIAN_WIDTH = 3 #cm
FLOOR_NUM = 3   #3
FLOOR_HEIGHT = 140  #cm
CAR_HEIGHT = 87 #cm
UAV_HEIGHT = 260    #cm 2层若以100高度飞应取125计算 3层对应260 4层对应400
H_CAMERA = FLOOR_NUM * FLOOR_HEIGHT - (CAR_HEIGHT + UAV_HEIGHT) - TIEPIAN_WIDTH #cm   将上方横梁的上边沿当作地面，参照论文把上下调转过来，此数值需要根据无人机的飞行高度、货架单层高度进行计算估计
# H_CAMERA = FLOOR_NUM * FLOOR_HEIGHT - (CAR_HEIGHT + UAV_HEIGHT) - TIEPIAN_WIDTH - RED_WIDTH

STANDARD_DEPTH = 88.67

BOXMPR_DEVICE = None
BOXMPR_MODEL = None

'''
    通过阈值暴力判断红色横梁的上边沿点
'''
def point_check(image, p_x, p_y):
    for y in range(p_y, 1, -1):
        if image[y, p_x] == 255 and image[y - 1, p_x] == 0:
            return y
    return 0

'''
    通过状态机的形式判断此时像素点位于横梁下、横梁中、横梁上
'''
def point_check(image, threshold, p_x, p_y):
    state = 0

    for y in range(p_y - threshold, threshold, -1):
        if state == 0:
            flag = True

            for i in range(y + threshold, y, -1):
                if image[i, p_x] != 0:
                    flag = False
                    break
            if not flag:
                continue

            for i in range(y, y - threshold, -1):
                if image[i, p_x] != 255:
                    flag = False
                    break
            if not flag:
                continue

            state = 1
        elif state == 1:
            flag = True

            for i in range(y + threshold, y, -1):
                if image[i, p_x] != 255:
                    flag = False
                    break
            if not flag:
                continue

            for i in range(y - 1, y - threshold, -1):
                if image[i, p_x] != 0:
                    flag = False
                    break
            if not flag:
                continue

            state = 2
            return y

    return 0

'''
    检测位于红色横梁上边沿的像素点坐标
'''
def up_red_side_detect(image, p0_x, p0_y, p1_x, p1_y):
    # 在彩色图像的情况下，解码图像将以b g r顺序存储通道。
    grid_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 从RGB色彩空间转换到HSV色彩空间
    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

    # H、S、V范围一：
    lower1 = np.array([0, 43, 46])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(grid_HSV, lower1, upper1)  # mask1 为二值图像
    res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)

    # H、S、V范围二：
    lower2 = np.array([156, 43, 46])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(grid_HSV, lower2, upper2)
    res2 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask2)

    # 将两个二值图像结果 相加
    mask3 = mask1 + mask2

    # 中值滤波滤除白色系带的干扰
    img_median_blur = cv2.medianBlur(mask3, 35) # param: 35

    # 孔洞填充去除标签的干扰
    contours, _ = cv2.findContours(img_median_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 30000:
            cv_contours.append(contour)
        else:
            continue
    cv2.fillPoly(img_median_blur, cv_contours, (255, 255, 255))

    # 对二值化图像进行膨胀腐蚀去除标签的干扰
    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))  # param:(35, 35)
    # erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (81, 81))  # param:(81, 81)
    # img_dilate = cv2.dilate(img_median_blur, dilate_kernel)
    # img_erosion = cv2.erode(img_dilate, erosion_kernel)

    # 根据黑白像素变化情况判断边沿点
    # 状态机：0->横梁下边沿以下 1->横梁 2->横梁上边沿以上
    p2_x = p0_x
    p2_y = 0
    p3_x = p1_x
    p3_y = 0

    # 左顶点
    p2_y = point_check(img_median_blur, 20, p0_x, p0_y) #10

    # 右顶点
    p3_y = point_check(img_median_blur, 20, p1_x, p1_y) #10

    # cv2.namedWindow('result', 0)
    # cv2.resizeWindow('result', 512, 512)
    # cv2.imshow('result', img_median_blur)
    # cv2.waitKey(0)

    return p2_x, p2_y, p3_x, p3_y, img_median_blur

'''
    透视变换矫正视角
'''
def parallelogram_to_rectangle(image, width, height, p0, p1, p2, p3):
    src_rect = np.array([p2, p3, p1, p0], dtype='float32')
    dst_rect = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]],
        dtype='float32')

    M = cv2.getPerspectiveTransform(src_rect, dst_rect)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

'''
    通过已知的横梁厚度基于二值化图像算出横梁与货物之间间隙的高度
'''
def gap_height_measurement_binary(image, width, height, red_width):
    # 对二值化图像进行腐蚀操作去除异常边界的干扰
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_erosion = cv2.erode(image.copy(), erosion_kernel)

    total_gap = 0
    for i in range(width):
        for j in range(height - 1, 0, -1):
            if img_erosion[j, i] == 0:
                total_gap = total_gap + 1
            # else:
            #     break
    total_red = 0
    for i in range(width):
        for j in range(height):
            if img_erosion[j, i] == 255:
                total_red = total_red + 1

    avg_gap = total_gap / width
    avg_red = total_red / width
    # avg_red = height - avg_gap
    print('avg_red: ', avg_red)
    print('avg_gap: ', avg_gap)
    gap_width = avg_gap * (red_width / avg_red)

    # cv2.namedWindow(image_name, 0)
    # cv2.resizeWindow(image_name, width, height)
    # cv2.imshow(image_name, img_erosion)
    # cv2.waitKey(0)
    return gap_width

'''
    Preprocess numpy image to torch tensor.
'''
def preprocess_image(image):
    if image.shape[0] != 512 or image.shape[1] != 512:
        image = cv2.resize(image, (512, 512))
    return torch.unsqueeze(ToTensor()(image), 0)

'''
    Given image read from opencv, return detected marking points.
'''
def detect_marking_points(detector, image, thresh, device):
    prediction = detector(preprocess_image(image).to(device))
    return get_predicted_points(prediction[0], thresh)

'''
    从推理的结果中获取角点坐标
'''
def get_cornerPoints(image, pred_points):
    height = image.shape[0]
    width = image.shape[1]
    points = []
    for confidence, marking_point in pred_points:
        p0_x = width * marking_point.x - 0.5
        p0_y = height * marking_point.y - 0.5
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        points.append([p0_x, p0_y])

    return points

'''
    利用检测器检测单张图片中的一对顶点
'''
def detect_image(detector, device, image):
    pred_points = detect_marking_points(detector, image, 0.05, device)  #0.05
    return get_cornerPoints(image, pred_points)

'''
    获取需要的设备和BoxMPR模型
'''
def get_device_and_BoxMPR_model():
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    torch.set_grad_enabled(False)
    dp_detector = DirectionalPointDetector(
        3, 32, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    dp_detector.load_state_dict(torch.load(BoxMPR_detector_weights, map_location=device))
    dp_detector.eval()

    return device, dp_detector

'''
    神经网络模型推理货物上端两顶点坐标
'''
def BoxMPR_inference(image):
    global BOXMPR_DEVICE, BOXMPR_MODEL
    # device, dp_detector = get_device_and_BoxMPR_model()
    if BOXMPR_DEVICE is None or BOXMPR_MODEL is None:
        BOXMPR_DEVICE, BOXMPR_MODEL = get_device_and_BoxMPR_model()

    return detect_image(BOXMPR_MODEL, BOXMPR_DEVICE, image)

'''
    筛选出成对的货物角点
'''
def points_filter(points, image):
    points.sort(key=lambda x: x[0], reverse=False)
    num_detected = len(points)
    point_pairs = []
    BOX_MAX_X_DIST = 600    #480
    BOX_MIN_X_DIST = 220
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
            point_i = points[i]
            point_j = points[j]
            # Step 1: length filtration.
            distance = calc_point_squre_dist(point_i, point_j)
            # 最早的数据适用
            # if distance < config.BOX_MIN_DIST or distance > config.BOX_MAX_DIST:
            #     continue
            # 20230729、20230812的数据并使用opencv拼接时适用
            # if distance < config.BOX_MIN_DIST or distance > 150000:
            #     continue
            # 20230926的数据适用
            if distance < 60000 or distance > 300000:   #60000,200000
                continue
            # 20230825的数据使用stitching_util拼接时适用
            # if distance < 16000 or distance > 58000:
            #     continue
            # Step 2: pass through filtration.
            if pass_through_third_point(points, i, j):
                continue
            # Step3: corner feature filtration.
            # if detect_area_harris_corner(point_i, point_j, image):
            #     continue

            # Step 4: corner left-right position filtration.
            if point_i[0] >= point_j[0]:
                continue

            # Step 5: corner vertical distance filtration
            if abs(point_i[1] - point_j[1]) >= config.BOX_MAX_VERTICAL_DIST:
                continue

            # Step 6: x length filtration.
            delta_x = abs(point_i[0] - point_j[0])
            if delta_x < config.BOX_MIN_X_DIST or delta_x > BOX_MAX_X_DIST:  # config.BOX_MIN_X_DIST
                continue

            # Step 7: bottom points filtration
            if point_i[1] >= 550 or point_j[1] >= 550:
                continue

            point_pairs.append((i, j))

    return point_pairs

'''
    滤除穿越点
'''
def pass_through_third_point(marking_points, i, j):
    """See whether the line between two points pass through a third point."""
    x_1 = marking_points[i][0]
    y_1 = marking_points[i][1]
    x_2 = marking_points[j][0]
    y_2 = marking_points[j][1]
    for point_idx, point in enumerate(marking_points):
        if point_idx == i or point_idx == j:
            continue
        x_0 = point[0]
        y_0 = point[1]
        vec1 = np.array([x_0 - x_1, y_0 - y_1])
        vec2 = np.array([x_2 - x_0, y_2 - y_0])
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        if np.dot(vec1, vec2) > config.SLOT_SUPPRESSION_DOT_PRODUCT_THRESH:
            return True
    return False

'''
    可视化调节置信度阈值的结果
'''
def threshold_test(points, image):
    for point in points:
        x = point[0]
        y = point[1]
        cv2.circle(image, (x, y), 5, (255, 0, 0), 3)

    cv2.namedWindow('test', 0)
    cv2.resizeWindow('test', 512, 512)
    cv2.imshow('test', image)
    cv2.waitKey(0)

'''
    获取关键位置的深度信息
'''
def key_area_depth(image_name, root_path, point1, point2, pillar_point1, pillar_point2):
    depth, width_factor, height_factor = depth_prediction(image_name, root_path)
    depth_height = depth.shape[0]
    depth_width = depth.shape[1]
    center_depth = depth[int(depth_height / 2)][int(depth_width / 2)]
    p1_depth, p2_depth, pillar_p1_depth, pillar_p2_depth = depth_estimation_operator(depth, width_factor, height_factor, point1, point2, pillar_point1, pillar_point2)

    print("box depth: " + str(center_depth))
    print("Detail depth info: ")
    print("p1_depth: " + str(p1_depth) + " p2_depth: " + str(p2_depth) + " pillar_p1_depth: " + str(pillar_p1_depth) + " pillar_p2_depth: " + str(pillar_p2_depth))

    return depth, p1_depth, p2_depth, pillar_p1_depth, pillar_p2_depth, center_depth

'''
    获取图片的深度信息
'''
def depth_prediction(image_name, root_path):
    output = depth_estimation(1, image_name, root_path)
    depth_width = output.shape[1]
    depth_height = output.shape[0]
    image = cv2.imread(os.path.join(root_path, image_name))
    image_width = image.shape[1]
    image_height = image.shape[0]

    width_factor = image_width / depth_width
    height_factor = image_height / depth_height
    return output, width_factor, height_factor

'''
    联合估计感兴趣区域的深度信息
'''
def depth_estimation_operator(depth, width_factor, height_factor, point1, point2, pillar_point1, pillar_point2):
    point1 = [int(point1[0] / width_factor), int(point1[1] / height_factor)]
    point2 = [int(point2[0] / width_factor), int(point2[1] / height_factor)]
    pillar_point1 = [int(pillar_point1[0] / width_factor), int(pillar_point1[1] / height_factor)]
    pillar_point2 = [int(pillar_point2[0] / width_factor), int(pillar_point2[1] / height_factor)]
    delta_x = abs(point2[0] - point1[0])

    offset_x = int(delta_x / 20)
    offset_y = offset_x
    print("offset: " + str(offset_x))
    sample_center1 = [point1[0] + offset_x, point1[1] + offset_y]
    sample_center2 = [point2[0] - offset_x, point2[1] + offset_y]
    sample_center3 = [pillar_point1[0] + offset_x, pillar_point1[1] + offset_y]
    sample_center4 = [pillar_point2[0] - offset_x, pillar_point2[1] + offset_y]
    grid_size = max(3, offset_x - 2)

    print("Sample center1: " + str(sample_center1[0]) + " - " + str(sample_center1[1]))
    print("Sample center2: " + str(sample_center2[0]) + " - " + str(sample_center2[1]))
    print("Sample center3: " + str(sample_center3[0]) + " - " + str(sample_center3[1]))
    print("Sample center4: " + str(sample_center4[0]) + " - " + str(sample_center4[1]))
    print("Depth shape: " + str(depth.shape[0]) + " - " + str(depth.shape[1]))

    # sample_area1 = np.array([[[sample_center1[0] - grid_size, sample_center1[1] - grid_size], [sample_center1[0], sample_center1[1] - grid_size], [sample_center1[0] + grid_size, sample_center1[1] - grid_size]],
    #                          [[sample_center1[0] - grid_size, sample_center1[1]], [sample_center1[0], sample_center1[1]], [sample_center1[0] + grid_size, sample_center1[1]]],
    #                          [[sample_center1[0] - grid_size, sample_center1[1] + grid_size], [sample_center1[0], sample_center1[1] + grid_size], [sample_center1[0] + grid_size, sample_center1[1] + grid_size]]])
    # sample_area2 = np.array([[[sample_center2[0] - grid_size, sample_center2[1] - grid_size], [sample_center2[0], sample_center2[1] - grid_size], [sample_center2[0] + grid_size, sample_center2[1] - grid_size]],
    #                          [[sample_center2[0] - grid_size, sample_center2[1]], [sample_center2[0], sample_center2[1]], [sample_center2[0] + grid_size, sample_center2[1]]],
    #                          [[sample_center2[0] - grid_size, sample_center2[1] + grid_size], [sample_center2[0], sample_center2[1] + grid_size], [sample_center2[0] + grid_size, sample_center2[1] + grid_size]]])

    sample_area1 = np.array([[[sample_center1[1] - grid_size, sample_center1[0] - grid_size], [sample_center1[1], sample_center1[0] - grid_size], [sample_center1[1] + grid_size, sample_center1[0] - grid_size]],
                             [[sample_center1[1] - grid_size, sample_center1[0]], [sample_center1[1], sample_center1[0]], [sample_center1[1] + grid_size, sample_center1[0]]],
                             [[sample_center1[1] - grid_size, sample_center1[0] + grid_size], [sample_center1[1], sample_center1[0] + grid_size], [sample_center1[1] + grid_size, sample_center1[0] + grid_size]]])
    sample_area2 = np.array([[[sample_center2[1] - grid_size, sample_center2[0] - grid_size], [sample_center2[1], sample_center2[0] - grid_size], [sample_center2[1] + grid_size, sample_center2[0] - grid_size]],
                             [[sample_center2[1] - grid_size, sample_center2[0]], [sample_center2[1], sample_center2[0]], [sample_center2[1] + grid_size, sample_center2[0]]],
                             [[sample_center2[1] - grid_size, sample_center2[0] + grid_size], [sample_center2[1], sample_center2[0] + grid_size], [sample_center2[1] + grid_size, sample_center2[0] + grid_size]]])

    p1_depth_arr = np.array([depth[sample_area1[0][0][0]][sample_area1[0][0][1]], depth[sample_area1[0][1][0]][sample_area1[0][1][1]], depth[sample_area1[0][2][0]][sample_area1[0][2][1]],
                             depth[sample_area1[1][0][0]][sample_area1[1][0][1]], depth[sample_area1[1][1][0]][sample_area1[1][1][1]], depth[sample_area1[1][2][0]][sample_area1[1][2][1]],
                             depth[sample_area1[2][0][0]][sample_area1[2][0][1]], depth[sample_area1[2][1][0]][sample_area1[2][1][1]], depth[sample_area1[2][2][0]][sample_area1[2][2][1]]])
    p2_depth_arr = np.array([depth[sample_area2[0][0][0]][sample_area2[0][0][1]], depth[sample_area2[0][1][0]][sample_area2[0][1][1]], depth[sample_area2[0][2][0]][sample_area2[0][2][1]],
                             depth[sample_area2[1][0][0]][sample_area2[1][0][1]], depth[sample_area2[1][1][0]][sample_area2[1][1][1]], depth[sample_area2[1][2][0]][sample_area2[1][2][1]],
                             depth[sample_area2[2][0][0]][sample_area2[2][0][1]], depth[sample_area2[2][1][0]][sample_area2[2][1][1]], depth[sample_area2[2][2][0]][sample_area2[2][2][1]]])

    # sample_area3 = np.array([[[sample_center3[0] - grid_size, sample_center3[1] - grid_size], [sample_center3[0], sample_center3[1] - grid_size], [sample_center3[0] + grid_size, sample_center3[1] - grid_size]],
    #                          [[sample_center3[0] - grid_size, sample_center3[1]], [sample_center3[0], sample_center3[1]], [sample_center3[0] + grid_size, sample_center3[1]]],
    #                          [[sample_center3[0] - grid_size, sample_center3[1] + grid_size], [sample_center3[0], sample_center3[1] + grid_size], [sample_center3[0] + grid_size, sample_center3[1] + grid_size]]])
    # sample_area4 = np.array([[[sample_center4[0] - grid_size, sample_center4[1] - grid_size], [sample_center4[0], sample_center4[1] - grid_size], [sample_center4[0] + grid_size, sample_center4[1] - grid_size]],
    #                          [[sample_center4[0] - grid_size, sample_center4[1]], [sample_center4[0], sample_center4[1]], [sample_center4[0] + grid_size, sample_center4[1]]],
    #                          [[sample_center4[0] - grid_size, sample_center4[1] + grid_size], [sample_center4[0], sample_center4[1] + grid_size], [sample_center4[0] + grid_size, sample_center4[1] + grid_size]]])
    sample_area3 = np.array([[[sample_center3[1] - grid_size, sample_center3[0] - grid_size], [sample_center3[1], sample_center3[0] - grid_size], [sample_center3[1] + grid_size, sample_center3[0] - grid_size]],
                             [[sample_center3[1] - grid_size, sample_center3[0]], [sample_center3[1], sample_center3[0]], [sample_center3[1] + grid_size, sample_center3[0]]],
                             [[sample_center3[1] - grid_size, sample_center3[0] + grid_size], [sample_center3[1], sample_center3[0] + grid_size], [sample_center3[1] + grid_size, sample_center3[0] + grid_size]]])
    sample_area4 = np.array([[[sample_center4[1] - grid_size, sample_center4[0] - grid_size], [sample_center4[1], sample_center4[0] - grid_size], [sample_center4[1] + grid_size, sample_center4[0] - grid_size]],
                             [[sample_center4[1] - grid_size, sample_center4[0]], [sample_center4[1], sample_center4[0]], [sample_center4[1] + grid_size, sample_center4[0]]],
                             [[sample_center4[1] - grid_size, sample_center4[0] + grid_size], [sample_center4[1], sample_center4[0] + grid_size], [sample_center4[1] + grid_size, sample_center4[0] + grid_size]]])

    pillar_p1_depth_arr = np.array([depth[sample_area3[0][0][0]][sample_area3[0][0][1]], depth[sample_area3[0][1][0]][sample_area3[0][1][1]], depth[sample_area3[0][2][0]][sample_area3[0][2][1]],
                                    depth[sample_area3[1][0][0]][sample_area3[1][0][1]], depth[sample_area3[1][1][0]][sample_area3[1][1][1]], depth[sample_area3[1][2][0]][sample_area3[1][2][1]],
                                    depth[sample_area3[2][0][0]][sample_area3[2][0][1]], depth[sample_area3[2][1][0]][sample_area3[2][1][1]], depth[sample_area3[2][2][0]][sample_area3[2][2][1]]])
    pillar_p2_depth_arr = np.array([depth[sample_area4[0][0][0]][sample_area4[0][0][1]], depth[sample_area4[0][1][0]][sample_area4[0][1][1]], depth[sample_area4[0][2][0]][sample_area4[0][2][1]],
                                    depth[sample_area4[1][0][0]][sample_area4[1][0][1]], depth[sample_area4[1][1][0]][sample_area4[1][1][1]], depth[sample_area4[1][2][0]][sample_area4[1][2][1]],
                                    depth[sample_area4[2][0][0]][sample_area4[2][0][1]], depth[sample_area4[2][1][0]][sample_area4[2][1][1]], depth[sample_area4[2][2][0]][sample_area4[2][2][1]]])

    p1_depth_arr = outliers_proc(p1_depth_arr)
    p2_depth_arr = outliers_proc(p2_depth_arr)
    pillar_p1_depth_arr = outliers_proc(pillar_p1_depth_arr)
    pillar_p2_depth_arr = outliers_proc(pillar_p2_depth_arr)
    # p1_depth = depth[sample_point1[0]][sample_point1[1]]
    # p2_depth = depth[sample_point2[0]][sample_point2[1]]

    return p1_depth_arr.mean(), p2_depth_arr.mean(), pillar_p1_depth_arr.mean(), pillar_p2_depth_arr.mean()

'''
    对离群深度值进行过滤处理
'''
def outliers_proc(data, scale=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    # print("Q1: " + str(Q1) + " Q3: " + str(Q3) + " IQR: " + str(IQR))
    data[data < Q1 - (scale * IQR)] = Q1 - (scale * IQR)
    data[data > Q3 + (scale * IQR)] = Q3 + (scale * IQR)

    return data

'''
    根据深度进行尺寸矫正计算
'''
def depth_correction(p1_depth, p2_depth, pillar_p1_depth, pillar_p2_depth, center_depth, gap_height):
    # print("Fig depth info: " + str(p1_depth) + ", " + str(p2_depth))
    # box_depth = max(p1_depth, p2_depth) if abs(p1_depth - p2_depth) >= 11 else (p1_depth + p2_depth) / 2
    box_depth = center_depth
    pillar_depth = (pillar_p1_depth + pillar_p2_depth) / 2
    delta_depth = STANDARD_DEPTH - box_depth
    print("Fig delta depth: " + str(delta_depth))

    gap_height = (1 + delta_depth / (2 * pillar_depth)) * gap_height
    # if abs(p1_depth - p2_depth) <= 15:
    #     gap_height = (1 + delta_depth / (2 * pillar_depth)) * gap_height
    # else:
    #     gap_height = fixed_error_correction(gap_height)

    return gap_height

'''
    BoxMPR主函数入口
'''
def BoxMPR_main(image_name):
    image = cv2.imread(os.path.join(image_dir, image_name))
    image = cv2.resize(image, (512, 512))
    height = image.shape[0]
    width = image.shape[1]

    points = BoxMPR_inference(image)
    # threshold_test(points, image)
    point_pairs = points_filter(points, image)
    p0_x = points[point_pairs[0][0]][0]
    p0_y = points[point_pairs[0][0]][1]
    p1_x = points[point_pairs[0][1]][0]
    p1_y = points[point_pairs[0][1]][1]

    if p0_x >= width or p0_x <= 0:
        print('P0 x position is invalid! P0 x position is set as half of width.')
        p0_x = width / 2
    if p0_y >= height or p0_y <= 0:
        print('P0 y position is invalid! P0 y position is set as half of height.')
        p0_y = height / 2
    if p1_x >= width or p1_x <= 0:
        print('P1 x position is invalid! P1 x position is set as half of width.')
        p1_x = width / 2
    if p1_y >= height or p1_y <= 0:
        print('P1 y position is invalid! P1 y position is set as half of height.')
        p1_y = height / 2

    radius = 5
    cv2.circle(image, (p0_x, p0_y), radius, (255, 0, 0), 3)
    cv2.circle(image, (p1_x, p1_y), radius, (0, 255, 0), 3)

    p2_x, p2_y, p3_x, p3_y, img_binary = up_red_side_detect(image, p0_x, p0_y, p1_x, p1_y)
    cv2.circle(image, (p2_x, p2_y), radius, (255, 255, 0), 3)
    cv2.circle(image, (p3_x, p3_y), radius, (0, 255, 255), 3)

    p0 = [p0_x, p0_y]  # 左下角
    p1 = [p1_x, p1_y]  # 右下角
    p2 = [p2_x, p2_y]  # 左上角
    p3 = [p3_x, p3_y]  # 右上角
    w = 560
    h = 240
    ROI = parallelogram_to_rectangle(img_binary, w, h, p0, p1, p2, p3)

    red_width = 9.8
    gap_height = gap_height_measurement_binary(ROI, w, h, red_width)
    print("Gap's height: ", gap_height)

    # cv2.imwrite(os.path.join(save_dir, image_name), image)
    cv2.namedWindow(image_name, 0)
    cv2.resizeWindow(image_name, 512, 512)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)

'''
    BoxMPR主函数入口
'''
def BoxMPR_LEDNet_main(image_dir, save_dir, image_name):
    image = cv2.imread(os.path.join(image_dir, image_name))
    height = image.shape[0]
    width = image.shape[1]
    points = BoxMPR_inference(image)
    if len(points) < 2:
        print("此处无货物")
        return -1, False

    # threshold_test(points, image)
    # print("Before filter: ", len(points))
    point_pairs = points_filter(points, image)
    # print("After filter: ", len(point_pairs))

    if (len(point_pairs) == 0):
        print("此处无货物")
        return -1, False

    if len(point_pairs[0]) != 2:
        print(image_name + " points number error! Points num: %d" % len(point_pairs[0]))
        return -1, False

    p0_x = points[point_pairs[0][0]][0]
    p0_y = points[point_pairs[0][0]][1]
    p1_x = points[point_pairs[0][1]][0]
    p1_y = points[point_pairs[0][1]][1]

    if p0_x >= width or p0_x <= 0:
        print('P0 x position is invalid! P0 x position is set as half of width.')
        p0_x = width / 2
    if p0_y >= height or p0_y <= 0:
        print('P0 y position is invalid! P0 y position is set as half of height.')
        p0_y = height / 2
    if p1_x >= width or p1_x <= 0:
        print('P1 x position is invalid! P1 x position is set as half of width.')
        p1_x = width / 2
    if p1_y >= height or p1_y <= 0:
        print('P1 y position is invalid! P1 y position is set as half of height.')
        p1_y = height / 2

    if abs(p0_y - p1_y) > 5:
        p0_y, p1_y = max(p0_y, p1_y), max(p0_y, p1_y)

    radius = 5
    cv2.circle(image, (p0_x, p0_y), radius, (255, 0, 0), 3)
    cv2.circle(image, (p1_x, p1_y), radius, (0, 255, 0), 3)

    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predict, mask = LEDNet_inference(RGB_image)
    # mask.save(os.path.join(r'F:\ProjectImagesDataset\IKEA\20230825\vertical_experiment\middle\lednet_result', image_name.replace('jpg', 'png')))
    # _, p2_y_hsv, _, p3_y_hsv, img_binary = up_red_side_detect(image, p0_x, p0_y, p1_x, p1_y)
    # _, p2_y_line, _, p3_y_line, image_line = upside_line_detect(image, predict, p0_x, p0_y, p1_x, p1_y)
    p2_y_line, p3_y_line = -1, -1
    # cv2.imwrite(os.path.join(r'F:\ProjectImagesDataset\IKEA\20230825\vertical_experiment\middle\line_result', image_name), image_line)
    p2_x, p2_y_lednet, p3_x, p3_y_lednet = upside_detect(predict, p0_x, p0_y, p1_x, p1_y)
    p2_y, p3_y = 0, 0
    if p2_y_line > 0 and p2_y_lednet > 0:
        p2_y = p2_y_line if p2_y_line < p2_y_lednet else p2_y_lednet
    if p3_y_line > 0 and p3_y_lednet > 0:
        p3_y = p3_y_line if p3_y_line < p3_y_lednet else p3_y_lednet
    if p2_y == 0:
        p2_y = p2_y_lednet if p2_y_lednet > 0 else p2_y_line
    if p3_y == 0:
        p3_y = p3_y_lednet if p3_y_lednet > 0 else p3_y_line

    if abs(p2_y - p3_y) > 5:
        p2_y, p3_y = min(p2_y, p3_y), min(p2_y, p3_y)

    cv2.circle(image, (p2_x, p2_y), radius, (255, 255, 0), 3)
    cv2.circle(image, (p3_x, p3_y), radius, (0, 255, 255), 3)

    p0 = [p0_x, p0_y]  # 左下角
    p1 = [p1_x, p1_y]  # 右下角
    p2 = [p2_x, p2_y]  # 左上角
    p3 = [p3_x, p3_y]  # 右上角
    # w = 560
    # h = 240
    w = int(((p3_x - p2_x) + (p1_x - p0_x)) / 2)
    h = int(((p1_y - p3_y) + (p0_y - p2_y)) / 2)
    # print("w: " + str(w) + " h: " + str(h))
    mask = np.array(mask)
    ROI = parallelogram_to_rectangle(mask, w, h, p0, p1, p2, p3)

    # cv2.namedWindow(image_name, 0)
    # cv2.resizeWindow(image_name, w, h)
    # cv2.imshow(image_name, ROI)
    # cv2.waitKey(0)

    # red_width = 9.8
    # gap_height = gap_height_measurement_mask(ROI, w, h, RED_WIDTH)
    gap_height = gap_height_measurement_based_on_camera_height(ROI, w, h, H_CAMERA, [p0_x, p0_y], [p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y])
    print("Gap's height: ", gap_height)

    # cv2.imwrite(os.path.join(save_dir, image_name.split('.')[0] + "_" + str(gap_height) + "." + image_name.split('.')[1]), image)
    cv2.imwrite(os.path.join(save_dir, 'vertical_' + image_name), image)
    return gap_height, True, [p0_x, p0_y], [p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]

def batch_process():
    df = pd.DataFrame(columns=("图片编号", "测量结果（单位：cm）"))
    for idx, file in enumerate(os.listdir(image_dir)):
        try:
            print("------""Processing file " + str(idx) + ": " + file + "......------")
            gap_height = BoxMPR_LEDNet_main(file)
            row_index = len(df) + 1  # 当前excel内容有几行
            df.loc[row_index] = [file, gap_height]
        except Exception as e:
            print(repr(e))

    df.to_excel(os.path.join(r'F:\ProjectImagesDataset\IKEA\20230825\vertical_experiment', "measurement.xlsx"), index=False)

def batch_process_test():
    total_processed = 0
    with open(r'F:\ProjectImagesDataset\IKEA\20230729\img_name.txt', 'r') as f:
        while True:
            img_name = f.readline()
            img_name = img_name.strip('\n')
            if not img_name:
                break
            if img_name.endswith("jpg"):
                total_processed += 1
                try:
                    print("------""Processing file: " + img_name + "......------")
                    gap_height = BoxMPR_LEDNet_main(img_name)
                except Exception as e:
                    print(repr(e))
    print("Total processed: ", total_processed)

def Serial_Images_Measurement(image_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file = get_file_description(save_dir, "measurement.txt")

    gap_height1, flag1 = -1, True
    gap_height3, flag3 = -1, True
    gap_height5, flag5 = -1, True
    try:
        gap_height1, flag1, fig1_point1, fig1_point2, fig1_pillar_point1, fig1_pillar_point2 = BoxMPR_LEDNet_main(image_dir, save_dir, '1.jpg')
        # print("fig1_point1: " + str(fig1_point1))
    except Exception as e:
        print("Fig1: " + repr(e))
        flag1 = False
    try:
        gap_height3, flag3, fig3_point1, fig3_point2, fig3_pillar_point1, fig3_pillar_point2 = BoxMPR_LEDNet_main(image_dir, save_dir, '3.jpg')
        # print("fig3_point1: " + str(fig3_point1))
    except Exception as e:
        print("Fig3: " + repr(e))
        flag3 = False
    try:
        gap_height5, flag5, fig5_point1, fig5_point2, fig5_pillar_point1, fig5_pillar_point2 = BoxMPR_LEDNet_main(image_dir, save_dir, '5.jpg')
        # print("fig5_point1: " + str(fig5_point1))
    except Exception as e:
        print("Fig5: " + repr(e))
        flag5 = False

    # 利用深度信息矫正尺度
    # if flag1:
    #     depth1, fig1_p1_depth, fig1_p2_depth, fig1_pillar_p1_depth, fig1_pillar_p2_depth, center_depth1 = key_area_depth('1.jpg', image_dir, fig1_point1, fig1_point2, fig1_pillar_point1, fig1_pillar_point2)
    #     gap_height1 = depth_correction(fig1_p1_depth, fig1_p2_depth, fig1_pillar_p1_depth, fig1_pillar_p2_depth, center_depth1, gap_height1)
    #     #   可视化深度信息
    #     plt.imshow(depth1)
    #     plt.savefig(os.path.join(save_dir, 'fig1_depth.jpg'))
    # if flag3:
    #     depth3, fig3_p1_depth, fig3_p2_depth, fig3_pillar_p1_depth, fig3_pillar_p2_depth, center_depth3 = key_area_depth('3.jpg', image_dir, fig3_point1, fig3_point2, fig3_pillar_point1, fig3_pillar_point2)
    #     gap_height3 = depth_correction(fig3_p1_depth, fig3_p2_depth, fig3_pillar_p1_depth, fig3_pillar_p2_depth, center_depth3, gap_height3)
    #     #   可视化深度信息
    #     plt.imshow(depth3)
    #     plt.savefig(os.path.join(save_dir, 'fig3_depth.jpg'))
    # if flag5:
    #     depth5, fig5_p1_depth, fig5_p2_depth, fig5_pillar_p1_depth, fig5_pillar_p2_depth, center_depth5 = key_area_depth('5.jpg', image_dir, fig5_point1, fig5_point2, fig5_pillar_point1, fig5_pillar_point2)
    #     gap_height5 = depth_correction(fig5_p1_depth, fig5_p2_depth, fig5_pillar_p1_depth, fig5_pillar_p2_depth, center_depth5, gap_height5)
    #     #   可视化深度信息
    #     plt.imshow(depth5)
    #     plt.savefig(os.path.join(save_dir, 'fig5_depth.jpg'))

    if flag1 and not flag3 and not flag5:
        print("纵向间隙尺寸\n间隙3:%.2f" % (gap_height1), file=file)
        print("库位中只放置了右侧货物", file=file)
        close_file_description(file)
        return

    if not flag1 and flag3 and not flag5:
        print("纵向间隙尺寸\n间隙2:%.2f" % (gap_height3), file=file)
        print("库位中只放置了中间货物", file=file)
        close_file_description(file)
        return

    if not flag1 and not flag3 and flag5:
        print("纵向间隙尺寸\n间隙1:%.2f" % (gap_height5), file=file)
        print("库位中只放置了左侧货物", file=file)
        close_file_description(file)
        return

    if flag1 and flag3 and not flag5:
        print("纵向间隙尺寸\n间隙2:%.2f\n间隙3:%.2f" % (gap_height3, gap_height1), file=file)
        print("库位中左侧无货物", file=file)
        close_file_description(file)
        return

    if flag1 and not flag3 and flag5:
        print("纵向间隙尺寸\n间隙1:%.2f\n间隙3:%.2f" % (gap_height5, gap_height1), file=file)
        print("库位中中间无货物", file=file)
        close_file_description(file)
        return

    if not flag1 and flag3 and flag5:
        print("纵向间隙尺寸\n间隙1:%.2f\n间隙2:%.2f" % (gap_height5, gap_height3), file=file)
        print("库位中右侧无货物", file=file)
        close_file_description(file)
        return

    if not flag1 and not flag3 and not flag5:
        print("库位中无货物，均处于安全距离", file=file)
        close_file_description(file)
        return

    print("纵向间隙尺寸\n间隙1:%.2f\n间隙2:%.2f\n间隙3:%.2f" % (gap_height5, gap_height3, gap_height1), file=file)
    close_file_description(file)

    return

'''
    批量完成图片的序列化测量
'''
def batch_serial_measurement(data_src_dir, data_dst_dir):
    dirs = os.listdir(data_src_dir)
    for dir in dirs:
        if dir.endswith('.txt'):
            continue
        print("Measuring " + dir + " vertical size......")
        try:
            Serial_Images_Measurement(os.path.join(data_src_dir, dir), os.path.join(data_dst_dir, dir))
        except Exception as e:
            f = get_file_description(data_dst_dir, 'fail_log.txt')
            f.write(dir + " vertical measurement fail! Please check this kuwei.")
            print("Exception info: " + repr(e), file=f)
            close_file_description(f)
    print("Measurement task complete!")

if __name__ == '__main__':
    # image_name = r'180_left.jpg'
    # BoxMPR_main(image_name)
    # BoxMPR_LEDNet_main(image_name)
    # batch_process()
    # batch_process_test()
    # Serial_Images_Measurement(image_dir, save_dir)
    batch_serial_measurement(data_src_dir, data_dst_dir)