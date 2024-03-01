# -*- coding: utf-8 -*-
# @Time    : 2024/1/15 16:39
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : line_detect.py
import cv2
import os
import numpy as np
import config
import torch
from model import DirectionalPointDetector
from torchvision.transforms import ToTensor
from data import get_predicted_points, calc_point_squre_dist
from semantic_info import LEDNet_inference, upside_detect
from line_fit import upside_line_detect, upside_line_detect_experiment_version

image_dir = r'C:\Users\95725\Desktop\line_detect\source'
save_dir = r'C:\Users\95725\Desktop\line_detect\filtered_lines'
BoxMPR_detector_weights = r'checkpoints\dp_detector_799_v100.pth'
LEDNet_detector_weights = r'checkpoints\LEDNet_iter_170400_v100.pth'

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
            point_pairs.append((i, j))

    return point_pairs

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
    pred_points = detect_marking_points(detector, image, 0.01, device)  #0.05
    return get_cornerPoints(image, pred_points)

'''
    神经网络模型推理货物上端两顶点坐标
'''
def BoxMPR_inference(image):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    torch.set_grad_enabled(False)
    dp_detector = DirectionalPointDetector(
        3, 32, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    dp_detector.load_state_dict(torch.load(BoxMPR_detector_weights, map_location=device))
    dp_detector.eval()
    return detect_image(dp_detector, device, image)

def detect_single_image(image_dir, image_name, save_dir):
    image = cv2.imread(os.path.join(image_dir, image_name))
    height = image.shape[0]
    width = image.shape[1]
    points = BoxMPR_inference(image)

    point_pairs = points_filter(points, image)

    p0_x = points[point_pairs[0][0]][0]
    p0_y = points[point_pairs[0][0]][1]
    p1_x = points[point_pairs[0][1]][0]
    p1_y = points[point_pairs[0][1]][1]

    if abs(p0_y - p1_y) > 5:
        p0_y, p1_y = max(p0_y, p1_y), max(p0_y, p1_y)

    # radius = 5
    # cv2.circle(image, (p0_x, p0_y), radius, (255, 0, 0), 3)
    # cv2.circle(image, (p1_x, p1_y), radius, (0, 255, 0), 3)

    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predict, mask = LEDNet_inference(RGB_image)
    # _, p2_y_line, _, p3_y_line, image_line = upside_line_detect(image, predict, p0_x, p0_y, p1_x, p1_y)
    _, p2_y_line, _, p3_y_line, image_line = upside_line_detect_experiment_version(image, predict, p0_x, p0_y, p1_x, p1_y, save_dir, image_name)
    # p2_y_line, p3_y_line = -1, -1
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

    if abs(p2_y - p3_y) > 10:
        p2_y, p3_y = min(p2_y, p3_y), min(p2_y, p3_y)

    cv2.line(image, (p0_x, p2_y), (p1_x, p3_y), (255, 0, 0), 4)
    cv2.imwrite(os.path.join(save_dir, image_name), image)


def batch_detect(image_dir, save_dir):
    image_names = os.listdir(image_dir)
    for image_name in image_names:
        try:
            detect_single_image(image_dir, image_name, save_dir)
        except Exception as e:
            print(str(e))

    print("Batch line detection complete!")

if __name__ == '__main__':
    batch_detect(image_dir, save_dir)