# -*- coding: utf-8 -*-
# @Time    : 2024/1/15 16:39
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : line_detect.py
import cv2
import os
from utils.semantic_info_util import LEDNet_inference, upside_detect
from utils.line_fit import upside_line_detect, upside_line_detect_experiment_version
from utils.marking_points_util import BoxMPR_inference, points_filter

image_dir = r'C:\Users\95725\Desktop\line_detect\source'
save_dir = r'C:\Users\95725\Desktop\line_detect\filtered_lines'


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