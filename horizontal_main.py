# -*- coding: utf-8 -*-
# @Time    : 2023/8/14 21:36
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : horizontal_main.py
import os
import cv2
import numpy as np
import config
import torch
import pandas as pd
from model import DirectionalPointDetector
from torchvision.transforms import ToTensor
from data import get_predicted_points, calc_point_squre_dist
from semantic_info import LEDNet_inference, upside_detect, gap_height_measurement_mask, pillar_detect, pillar_detect_partial
from line_fit import upside_line_detect
from utils import stitch_image, opencv_stitch, get_file_description, close_file_description
from stitching import Stitcher

# image_dir = r'D:\ProjectCodes\VisionMeasurement\stiching_test\img_dir\kuwei6'
# save_dir = r'D:\ProjectCodes\VisionMeasurement\stiching_test\result_dir\kuwei6'
image_dir = r'C:\Users\95725\Desktop\src\kuwei15'
save_dir = r'C:\Users\95725\Desktop\dst\kuwei15'
data_src_dir = r'C:\Users\95725\Desktop\src'
data_dst_dir = r'C:\Users\95725\Desktop\dst'
# image_dir = r'D:\ProjectCodes\VisionMeasurement\1020test\floor4\src\kuwei1'
# save_dir = r'D:\ProjectCodes\VisionMeasurement\1020test\floor4\dst\kuwei1'
image_name = '1.jpg'
BoxMPR_detector_weights = r'checkpoints\dp_detector_59_dark.pth'   #r'checkpoints\dp_detector_799_v100.pth'
LEDNet_detector_weights = r'checkpoints\LEDNet_iter_170400_v100.pth'
CROP = 1
OPERATOR = "sift"
CONFIDENCE = 0.2
LEFT_OFFSET = 1.6   #cm
RIGHT_OFFSET = -1.4 #cm
LEFT_CENTER_OFFSET = 0.1    #cm
RIGHT_CENTER_OFFSET = 0.9   #cm
PILLAR_WIDTH = 11.8 #cm
KUWEI_WIDTH = 325   #cm

'''
    依次读取目录下的图片
'''
def partial_image(image_dir):
    img_list = []
    # delta = 512
    # for i in range(3):
    #     img_list.append(stitched_img[0:512, (delta*i):(delta*i + delta)])
    for img_name in os.listdir(image_dir):
        img_list.append(cv2.imread(os.path.join(image_dir, img_name)))

    return img_list


'''
    对全景图中的三个货物分别进行采样
'''
def sample_stitched(stitched_img):
    img_list = []
    offset_x = []
    size_x = []
    height = stitched_img.shape[0]
    width = stitched_img.shape[1]

    #img1
    start_ratio, end_ratio = 0.05, 0.4
    sample1 = stitched_img[0:height, int(start_ratio * width):int(end_ratio * width)]
    sample1_width = int(end_ratio * width) - int(start_ratio * width)
    img_list.append(cv2.resize(sample1, (512, 512)))
    offset_x.append(int(start_ratio * width))
    size_x.append(sample1_width)

    # img2
    start_ratio, end_ratio = 0.25, 0.75
    sample2 = stitched_img[0:height, int(start_ratio * width):int(end_ratio * width)]
    sample2_width = int(end_ratio * width) - int(start_ratio * width)
    img_list.append(cv2.resize(sample2, (512, 512)))
    offset_x.append(int(start_ratio * width))
    size_x.append(sample2_width)

    # img3
    start_ratio, end_ratio = 0.6, 0.95
    sample3 = stitched_img[0:height, int(start_ratio * width):int(end_ratio * width)]
    sample3_width = int(end_ratio * width) - int(start_ratio * width)
    img_list.append(cv2.resize(sample3, (512, 512)))
    offset_x.append(int(start_ratio * width))
    size_x.append(sample3_width)

    return img_list, offset_x, size_x, height, width


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
    pred_points = detect_marking_points(detector, image, 0.05, device)  #threshold:0.05
    return get_cornerPoints(image, pred_points)

'''
    筛选出成对的货物角点
'''
def points_filter(points, image):
    paired_points = []
    num_detected = len(points)
    point_pairs = []
    BOX_MAX_X_DIST = 600    #480
    BOX_MIN_X_DIST = 300
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
            if paired_points.__contains__(i) or paired_points.__contains__(j):
                continue

            point_i = points[i]
            point_j = points[j]
            # Step 1: length filtration.
            distance = calc_point_squre_dist(point_i, point_j)
            # print(distance)
            # 最早的数据适用
            # if distance < config.BOX_MIN_DIST or distance > config.BOX_MAX_DIST:
            #     continue
            # 20230729、20230812的数据并使用opencv拼接时适用
            # if distance < config.BOX_MIN_DIST or distance > 150000:
            #     continue
            # 20230926的数据适用
            if distance < 60000 or distance > 300000:   #300000
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
            if delta_x < BOX_MIN_X_DIST or delta_x > BOX_MAX_X_DIST: # config.BOX_MIN_X_DIST
                continue
            point_pairs.append((i, j))
            paired_points.append(i);
            paired_points.append(j);

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
            # print("i: " + str(i) + " j: " + str(j))
            # print("point idx: ", point_idx)
            # print(str(np.dot(vec1, vec2)))
            return True
    return False

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

'''
    批量预测图片序列中的货物角点
'''
def predict_img_list(img_list):
    p0_list = []
    p1_list = []
    for idx in range(len(img_list)):
        image = img_list[idx]
        image = cv2.resize(image, (512, 512))
        height = image.shape[0]
        width = image.shape[1]
        points = BoxMPR_inference(image)
        point_pairs = points_filter(points, image)
        p0_x = points[point_pairs[0][0]][0]
        p0_y = points[point_pairs[0][0]][1]
        p1_x = points[point_pairs[0][1]][0]
        p1_y = points[point_pairs[0][1]][1]
        points_0 = np.array([p0_x, p0_y, 1])
        points_1 = np.array([p1_x, p1_y, 1])
        p0_list.append(points_0)
        p1_list.append(points_1)
        # cv2.namedWindow("partial_img", 0)
        # cv2.imshow("partial_img", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return p0_list, p1_list

'''
    根据单应性矩阵完成角点坐标转换及标准化
'''
def transform_normalize_points(p0_list, p1_list, H_list):
    transformed_point0_list = []
    transformed_point1_list = []
    for idx in range(len(p0_list)):
        origin_point0 = p0_list[idx].T
        origin_point1 = p1_list[idx].T
        for h_idx in range(len(H_list) - idx):
            origin_point0 = H_list[h_idx].dot(origin_point0)
            origin_point1 = H_list[h_idx].dot(origin_point1)

        transformed_point0_list.append(origin_point0.T)
        transformed_point1_list.append(origin_point1.T)

    '''
        以z轴坐标为基准进行标准化，使坐标z=1
    '''
    point0_list = []
    point1_list = []
    for point in transformed_point0_list:
        point = point / point[2]
        point0_list.append(point)
    for point in transformed_point1_list:
        point = point / point[2]
        point1_list.append(point)

    return point0_list, point1_list

'''
    按比例进行横向尺寸测量
'''
def scale_measurement(key_point_list):
    pillar_1 = key_point_list[1][0] - key_point_list[0][0]
    pillar_2 = key_point_list[len(key_point_list) - 1][0] - key_point_list[len(key_point_list) - 2][0]
    pillar_pixel_width = (pillar_1 + pillar_2) / 2

    kuwei_pixel_width = key_point_list[9][0] - key_point_list[0][0]

    horizontal_pixel_gap = []
    for idx in range(1, len(key_point_list) - 1, 2):
        horizontal_pixel_gap.append(key_point_list[idx + 1][0] - key_point_list[idx][0])
    box_pixel_width = []
    for idx in range(2, len(key_point_list) - 2, 2):
        box_pixel_width.append(key_point_list[idx + 1][0] - key_point_list[idx][0])

    # horizontal_gap = horizontal_pixel_gap / pillar_pixel_width * PILLAR_WIDTH
    # box_width = box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
    horizontal_gap = horizontal_pixel_gap / kuwei_pixel_width * KUWEI_WIDTH
    box_width = box_pixel_width / kuwei_pixel_width * KUWEI_WIDTH
    print("从左至右间隙宽度(cm): ", horizontal_gap)
    print("从左至右货物宽度(cm): ", box_width)

    return horizontal_gap, box_width

'''
    根据图片序列按比例进行横向尺寸测量
'''
def serial_scale_measurement(key_point_list):
    pillar_1 = key_point_list[1][0] - key_point_list[0][0]
    pillar_2 = key_point_list[len(key_point_list) - 1][0] - key_point_list[len(key_point_list) - 2][0]
    pillar_pixel_width = (pillar_1 + pillar_2) / 2

    kuwei_pixel_width = key_point_list[9][0] - key_point_list[0][0]

    horizontal_pixel_gap = []
    for idx in range(1, len(key_point_list) - 1, 2):
        horizontal_pixel_gap.append(key_point_list[idx + 1][0] - key_point_list[idx][0])
    box_pixel_width = []
    for idx in range(2, len(key_point_list) - 2, 2):
        box_pixel_width.append(key_point_list[idx + 1][0] - key_point_list[idx][0])

    # horizontal_gap = horizontal_pixel_gap / pillar_pixel_width * PILLAR_WIDTH
    # box_width = box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
    horizontal_gap = horizontal_pixel_gap / kuwei_pixel_width * KUWEI_WIDTH
    box_width = box_pixel_width / kuwei_pixel_width * KUWEI_WIDTH
    print("从左至右间隙宽度(cm): ", horizontal_gap)
    print("从左至右货物宽度(cm): ", box_width)

    return horizontal_gap, box_width

'''
    按比例进行横向尺寸测量（分为左右两半各一张图片分别测量）
'''
def scale_measurement_partial(key_point_list, direction):
    if direction == 0:
        pillar_pixel_width = key_point_list[1][0] - key_point_list[0][0]
        horizontal_pixel_gap = []
        for idx in range(1, len(key_point_list) - 1, 2):
            horizontal_pixel_gap.append(key_point_list[idx + 1][0] - key_point_list[idx][0])
        box_pixel_width = []
        for idx in range(2, len(key_point_list) - 1, 2):
            box_pixel_width.append(key_point_list[idx + 1][0] - key_point_list[idx][0])
    elif direction == 1:
        pillar_pixel_width = key_point_list[len(key_point_list) - 1][0] - key_point_list[len(key_point_list) - 2][0]
        horizontal_pixel_gap = []
        for idx in range(len(key_point_list) - 2, 0, -2):
            horizontal_pixel_gap.append(key_point_list[idx][0] - key_point_list[idx - 1][0])
        box_pixel_width = []
        for idx in range(len(key_point_list) - 3, -1, -2):
            box_pixel_width.append(key_point_list[idx][0] - key_point_list[idx - 1][0])

    horizontal_gap = horizontal_pixel_gap / pillar_pixel_width * PILLAR_WIDTH
    box_width = box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
    print("从左至右间隙宽度(cm): ", horizontal_gap)
    print("从左至右货物宽度(cm): ", box_width)


    return horizontal_gap, box_width


'''
    对标记点进行识别并将数据结构调整为方便处理的形式
'''
def points_extractor(image_dir, img_name, point_num):
    img = cv2.imread(os.path.join(image_dir, img_name))
    points = BoxMPR_inference(img)
    if (point_num == 2 and len(points) < 2) or (point_num == 4 and len(points) < 4):
        print("length of points: ", len(points))
        print("此处无货物")
        return points, img, False
    points.sort(key=lambda x: x[0], reverse=False)
    if point_num == 2:
        point_pairs = points_filter(points, img)
        if(len(point_pairs) == 0):
            print("此处无货物")
            return points, img, False

        if len(point_pairs) != 1:
            print(img_name + " points number error! Points num: %d" % len(points))
            print(img_name + " points number error! Point pairs num: %d" % len(point_pairs))
            return points, img, False

        p0_x = points[point_pairs[0][0]][0]
        p0_y = points[point_pairs[0][0]][1]
        p1_x = points[point_pairs[0][1]][0]
        p1_y = points[point_pairs[0][1]][1]
        if abs(p0_y - p1_y) > 5:
            p0_y, p1_y = max(p0_y, p1_y), max(p0_y, p1_y)
        points = []
        points.append([p0_x, p0_y])
        points.append([p1_x, p1_y])
    elif point_num == 4:
        # for point in points:
        #     print("x: " + str(point[0]) + " y: " + str(point[1]))

        # if len(points) != 4:
        #     print(img_name + " points number error! Points num: %d" % len(points))
        #     return points, img, False

        point_pairs = points_filter(points, img)
        if len(point_pairs) != 2:
            print(img_name + " points number error! Points num: %d" % len(points))
            print(img_name + " points number error! Point pairs num: %d" % len(point_pairs))
            return points, img, False

        p0_x = points[point_pairs[0][0]][0]
        p0_y = points[point_pairs[0][0]][1]
        p1_x = points[point_pairs[0][1]][0]
        p1_y = points[point_pairs[0][1]][1]
        if abs(p0_y - p1_y) > 5:
            p0_y, p1_y = max(p0_y, p1_y), max(p0_y, p1_y)
        p2_x = points[point_pairs[1][0]][0]
        p2_y = points[point_pairs[1][0]][1]
        p3_x = points[point_pairs[1][1]][0]
        p3_y = points[point_pairs[1][1]][1]
        if abs(p2_y - p3_y) > 5:
            p2_y, p3_y = max(p2_y, p3_y), max(p2_y, p3_y)
        points = []
        points.append([p0_x, p0_y])
        points.append([p1_x, p1_y])
        points.append([p2_x, p2_y])
        points.append([p3_x, p3_y])

    return points, img, True

'''
    使用手写的拼接方法进行全景图恢复
'''
def Stitching_Util_main(image_dir):
    stitched_img, H_list = stitch_image(image_dir)
    img_list = partial_image(stitched_img, image_dir)
    # cv2.imwrite(os.path.join(image_dir, "stitched_img.jpg"), stitched_img)

    p0_list, p1_list = predict_img_list(img_list)

    point0_list, point1_list = transform_normalize_points(p0_list, p1_list, H_list)

    print(point0_list)
    print(point1_list)
    height = stitched_img.shape[0]
    width = stitched_img.shape[1]

    radius = 5
    for idx in range(len(point0_list)):
        p0_x, p0_y = int(point0_list[idx][0]), int(point0_list[idx][1])
        p1_x, p1_y = int(point1_list[idx][0]), int(point1_list[idx][1])
        cv2.circle(stitched_img, (p0_x, p0_y), radius, (255, 0, 0), 3)
        cv2.circle(stitched_img, (p1_x, p1_y), radius, (0, 255, 0), 3)

    cv2.namedWindow("stitched_img", 0)
    cv2.imshow("stitched_img", stitched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
    使用PyPI的stitching库进行全景图恢复
'''
def Stitching_main(image_dir):
    # 拼接图像 toDo:拼接结束后继续执行其他代码会抛出异常，要考虑如何处理。拼接只能从右往左进行
    stitcher = Stitcher(detector=OPERATOR, confidence_threshold=CONFIDENCE, matches_graph_dot_file=True)    #detector="sift", confidence_threshold=0.2
    images = []
    for img in os.listdir(image_dir):
        images.append(cv2.imread(os.path.join(image_dir, img)))
    panorama = stitcher.stitch(images)
    cv2.imwrite(os.path.join(save_dir, "panoroma.jpg"), panorama)

    panorama = cv2.imread(os.path.join(save_dir, "panoroma.jpg"))
    img_list, offset_x, size_x, height, width = sample_stitched(panorama)
    p0_list, p1_list = [], []
    for idx, img in enumerate(img_list):
        print("Inference sample: ", idx)
        cv2.imwrite(os.path.join(save_dir, "sample" + str(idx) + ".jpg"), img)

        points = BoxMPR_inference(img)
        point_pairs = points_filter(points, img)
        if len(point_pairs) == 0:
            continue
        p0_x = points[point_pairs[0][0]][0]
        p0_y = points[point_pairs[0][0]][1]
        p1_x = points[point_pairs[0][1]][0]
        p1_y = points[point_pairs[0][1]][1]

        p0_x = p0_x / 512 * size_x[idx] + offset_x[idx]
        p0_y = p0_y / 512 * height
        p1_x = p1_x / 512 * size_x[idx] + offset_x[idx]
        p1_y = p1_y / 512 * height

        points_0 = np.array([p0_x, p0_y])
        points_1 = np.array([p1_x, p1_y])
        p0_list.append(points_0)
        p1_list.append(points_1)

    RGB_image = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    predict, mask = LEDNet_inference(RGB_image)

    radius = 5
    y_line = 0
    corner_point_list = []
    for idx in range(len(p0_list)):
        p0_x, p0_y = int(p0_list[idx][0]), int(p0_list[idx][1])
        p1_x, p1_y = int(p1_list[idx][0]), int(p1_list[idx][1])
        corner_point_list.append(np.array([p0_x, p0_y]))
        corner_point_list.append(np.array([p1_x, p1_y]))
        y_line = max(y_line, max(p0_y, p1_y))
        cv2.circle(panorama, (p0_x, p0_y), radius, (255, 0, 0), 3)
        cv2.circle(panorama, (p1_x, p1_y), radius, (0, 255, 0), 3)

    left_pillar_left_x, left_pillar_right_x, right_pillar_left_x, right_pillar_right_x = pillar_detect(predict, int(
        p0_list[0][0]), int(p1_list[len(p0_list) - 1][0]), y_line, width)

    key_point_list = []
    key_point_list.append(np.array([left_pillar_left_x, y_line]))
    key_point_list.append(np.array([left_pillar_right_x, y_line]))
    for point in corner_point_list:
        key_point_list.append(point)
    key_point_list.append(np.array([right_pillar_left_x, y_line]))
    key_point_list.append(np.array([right_pillar_right_x, y_line]))

    cv2.circle(panorama, (left_pillar_left_x, y_line), radius, (0, 0, 255), 3)
    cv2.circle(panorama, (left_pillar_right_x, y_line), radius, (0, 0, 255), 3)
    cv2.circle(panorama, (right_pillar_left_x, y_line), radius, (0, 0, 255), 3)
    cv2.circle(panorama, (right_pillar_right_x, y_line), radius, (0, 0, 255), 3)

    print("key point list length: ", len(key_point_list))
    # horizontal_gap, box_width = scale_measurement(key_point_list)

    cv2.imwrite(os.path.join(save_dir, "stitched_marking_point.jpg"), panorama)
    cv2.namedWindow("stitched_img", 0)
    cv2.imshow("stitched_img", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
    使用Opencv的拼接方法进行全景图恢复
'''
def Opencv_Stitching_main(image_dir):
    stitched_img = opencv_stitch(image_dir, CROP)
    cv2.imwrite(os.path.join(save_dir, "stitched_raw.jpg"), stitched_img)
    img_list, offset_x, size_x, height, width = sample_stitched(stitched_img)
    p0_list, p1_list = [], []
    for idx, img in enumerate(img_list):
        print("Inference sample: ", idx)
        cv2.imwrite(os.path.join(save_dir, "sample" +  str(idx) + ".jpg"), img)
        # cv2.namedWindow("sample_img", 0)
        # cv2.imshow("sample_img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        points = BoxMPR_inference(img)
        point_pairs = points_filter(points, img)
        if len(point_pairs) == 0:
            continue
        p0_x = points[point_pairs[0][0]][0]
        p0_y = points[point_pairs[0][0]][1]
        p1_x = points[point_pairs[0][1]][0]
        p1_y = points[point_pairs[0][1]][1]

        p0_x = p0_x / 512 * size_x[idx] + offset_x[idx]
        p0_y = p0_y / 512 * height
        p1_x = p1_x / 512 * size_x[idx] + offset_x[idx]
        p1_y = p1_y / 512 * height

        points_0 = np.array([p0_x, p0_y])
        points_1 = np.array([p1_x, p1_y])
        p0_list.append(points_0)
        p1_list.append(points_1)

    RGB_image = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
    predict, mask = LEDNet_inference(RGB_image)

    radius = 5
    y_line = 0
    corner_point_list = []
    for idx in range(len(p0_list)):
        p0_x, p0_y = int(p0_list[idx][0]), int(p0_list[idx][1])
        p1_x, p1_y = int(p1_list[idx][0]), int(p1_list[idx][1])
        corner_point_list.append(np.array([p0_x, p0_y]))
        corner_point_list.append(np.array([p1_x, p1_y]))
        y_line = max(y_line, max(p0_y, p1_y))
        cv2.circle(stitched_img, (p0_x, p0_y), radius, (255, 0, 0), 3)
        cv2.circle(stitched_img, (p1_x, p1_y), radius, (0, 255, 0), 3)

    left_pillar_left_x, left_pillar_right_x, right_pillar_left_x, right_pillar_right_x = pillar_detect(predict, int(p0_list[0][0]), int(p1_list[len(p0_list) - 1][0]), y_line, width)

    key_point_list = []
    key_point_list.append(np.array([left_pillar_left_x, y_line]))
    key_point_list.append(np.array([left_pillar_right_x, y_line]))
    for point in corner_point_list:
        key_point_list.append(point)
    key_point_list.append(np.array([right_pillar_left_x, y_line]))
    key_point_list.append(np.array([right_pillar_right_x, y_line]))

    cv2.circle(stitched_img, (left_pillar_left_x, y_line), radius, (0, 0, 255), 3)
    cv2.circle(stitched_img, (left_pillar_right_x, y_line), radius, (0, 0, 255), 3)
    cv2.circle(stitched_img, (right_pillar_left_x, y_line), radius, (0, 0, 255), 3)
    cv2.circle(stitched_img, (right_pillar_right_x, y_line), radius, (0, 0, 255), 3)

    horizontal_gap, box_width = scale_measurement(key_point_list)

    cv2.imwrite(os.path.join(save_dir, "stitched_marking_point.jpg"), stitched_img)
    cv2.namedWindow("stitched_img", 0)
    cv2.imshow("stitched_img", stitched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
    对一张图像分为左右两部分进行测量
'''
def Compose_Measurement(image_dir):
    img = cv2.imread(os.path.join(image_dir, image_name))
    points = BoxMPR_inference(img)
    points.sort(key = lambda x:x[0], reverse=False)

    if len(points) is not 4:
        print("Points number error!")
        return

    RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predict, mask = LEDNet_inference(RGB_image)

    radius = 5
    y_line = 0
    left_x, right_x = 1024, 0
    for point in points:
        y_line = max(y_line, int(point[1]))
        left_x = min(left_x, int(point[0]))
        right_x = max(right_x, int(point[0]))
        cv2.circle(img, (point[0], point[1]), radius, (0, 255, 0), 3)

    # 左半部分测量
    # left_pillar_left_x, left_pillar_right_x = pillar_detect_partial(predict, left_x, y_line, img.shape[1], 0)
    # key_point_list = []
    # key_point_list.append(np.array([left_pillar_left_x, y_line]))
    # key_point_list.append(np.array([left_pillar_right_x, y_line]))
    # for point in points:
    #     key_point_list.append(point)
    #
    # cv2.circle(img, (left_pillar_left_x, y_line), radius, (0, 0, 255), 3)
    # cv2.circle(img, (left_pillar_right_x, y_line), radius, (0, 0, 255), 3)
    # horizontal_gap, box_width = scale_measurement_partial(key_point_list, 0)

    # 右半部分测量
    right_pillar_left_x, right_pillar_right_x = pillar_detect_partial(predict, right_x, y_line, img.shape[1], 1)
    key_point_list = []
    for point in points:
        key_point_list.append(point)
    key_point_list.append(np.array([right_pillar_left_x, y_line]))
    key_point_list.append(np.array([right_pillar_right_x, y_line]))

    cv2.circle(img, (right_pillar_left_x, y_line), radius, (0, 0, 255), 3)
    cv2.circle(img, (right_pillar_right_x, y_line), radius, (0, 0, 255), 3)
    horizontal_gap, box_width = scale_measurement_partial(key_point_list, 1)


    cv2.imwrite(os.path.join(save_dir, "marking_point2.jpg"), img)
    cv2.namedWindow("target", 0)
    cv2.imshow("target", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
    对一个库位的序列化图片进行测量
'''
def Serial_Images_Measurement(image_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file = get_file_description(save_dir, "measurement.txt")

    points1, img1, flag1 = points_extractor(image_dir, '1.jpg', 2)
    points2, img2, flag2 = points_extractor(image_dir, '2.jpg', 4)
    points3, img3, flag3 = points_extractor(image_dir, '3.jpg', 2)
    points4, img4, flag4 = points_extractor(image_dir, '4.jpg', 4)
    points5, img5, flag5 = points_extractor(image_dir, '5.jpg', 2)

    if not flag1 and not flag2 and not flag3 and not flag4 and not flag5:
        print("库位中无货物，均处于安全距离", file=file)
        close_file_description(file)
        return

    if flag1:
        RGB_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        predict1, mask1 = LEDNet_inference(RGB_image1)

    if flag5:
        RGB_image5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
        predict5, mask5 = LEDNet_inference(RGB_image5)

    radius = 5

    y_line_1 = 0
    left_x_1, right_x_1 = 1024, 0
    if flag1:
        for point in points1:
            y_line_1 = max(y_line_1, int(point[1]))
            left_x_1 = min(left_x_1, int(point[0]))
            right_x_1 = max(right_x_1, int(point[0]))
            cv2.circle(img1, (point[0], point[1]), radius, (0, 255, 0), 3)

    # cv2.imwrite(os.path.join(save_dir, "marking_point2.jpg"), img1)
    # cv2.namedWindow("img1", 0)
    # cv2.imshow("img1", img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if flag2:
        for point in points2:
            cv2.circle(img2, (point[0], point[1]), radius, (0, 255, 0), 3)

    cv2.imwrite(os.path.join(save_dir, "img2.jpg"), img2)
    # cv2.namedWindow("img2", 0)
    # cv2.imshow("img2", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if flag3:
        for point in points3:
            cv2.circle(img3, (point[0], point[1]), radius, (0, 255, 0), 3)

    cv2.imwrite(os.path.join(save_dir, "img3.jpg"), img3)
    # cv2.namedWindow("img3", 0)
    # cv2.imshow("img3", img3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if flag4:
        for point in points4:
            cv2.circle(img4, (point[0], point[1]), radius, (0, 255, 0), 3)

    cv2.imwrite(os.path.join(save_dir, "img4.jpg"), img4)
    # cv2.namedWindow("img4", 0)
    # cv2.imshow("img4", img4)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    y_line_5 = 0
    left_x_5, right_x_5 = 1024, 0
    if flag5:
        for point in points5:
            y_line_5 = max(y_line_5, int(point[1]))
            left_x_5 = min(left_x_5, int(point[0]))
            right_x_5 = max(right_x_5, int(point[0]))
            cv2.circle(img5, (point[0], point[1]), radius, (0, 255, 0), 3)

    # cv2.imwrite(os.path.join(save_dir, "marking_point2.jpg"), img1)
    # cv2.namedWindow("img5", 0)
    # cv2.imshow("img5", img5)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 左半部分测量
    if flag5:
        left_pillar_left_x, left_pillar_right_x = pillar_detect_partial(predict5, left_x_5 + 20, int(img5.shape[0] / 2), img5.shape[1], 0)
        cv2.circle(img5, (left_pillar_left_x, y_line_5), radius, (0, 0, 255), 3)
        cv2.circle(img5, (left_pillar_right_x, y_line_5), radius, (0, 0, 255), 3)

    cv2.imwrite(os.path.join(save_dir, "img5.jpg"), img5)
    # cv2.namedWindow("img5", 0)
    # cv2.imshow("img5", img5)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 右半部分测量
    if flag1:
        right_pillar_left_x, right_pillar_right_x = pillar_detect_partial(predict1, right_x_1 - 20, int(img1.shape[0] / 2), img1.shape[1], 1)
        cv2.circle(img1, (right_pillar_left_x, y_line_1), radius, (0, 0, 255), 3)
        cv2.circle(img1, (right_pillar_right_x, y_line_1), radius, (0, 0, 255), 3)

    cv2.imwrite(os.path.join(save_dir, "img1.jpg"), img1)
    # cv2.namedWindow("img1", 0)
    # cv2.imshow("img1", img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if flag3 and not flag1 and not flag5:
        print("库位中只存在中间位置的货物，两侧不存在货物，处于安全距离。", file=file)
        close_file_description(file)
        return

    if flag5 and not flag3 and not flag1:
        left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
        left_box_pixel_width = points5[1][0] - points5[0][0]
        left_pillar_box_gap_pixel_width = points5[0][0] - left_pillar_right_x
        pillar_pixel_width = left_pillar_pixel_width
        left_box_width = left_box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        print("横向间隙尺寸\n间隙1:%.2f" % (left_pillar_box_gap_width + LEFT_OFFSET), file=file)
        print("横向货物尺寸\n货物1:%.2f" % (left_box_width), file=file)
        print("库位中只存在左侧位置的货物，其他位置不存在货物，右侧处于安全距离。", file=file)
        close_file_description(file)
        return

    if flag1 and not flag3 and not flag5:
        right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
        right_box_pixel_width = points1[1][0] - points1[0][0]
        right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]
        pillar_pixel_width = right_pillar_pixel_width
        right_box_width = right_box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        print("横向间隙尺寸\n间隙4:%.2f" % (right_pillar_box_gap_width + RIGHT_OFFSET), file=file)
        print("横向货物尺寸\n货物3:%.2f" % (right_box_width), file=file)
        print("库位中只存在右侧位置的货物，其他位置不存在货物，左侧处于安全距离。", file=file)
        close_file_description(file)
        return

    if flag5 and flag1 and not flag3:
        left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
        left_box_pixel_width = points5[1][0] - points5[0][0]
        left_pillar_box_gap_pixel_width = points5[0][0] - left_pillar_right_x
        right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
        right_box_pixel_width = points1[1][0] - points1[0][0]
        right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]

        pillar_pixel_width = (left_pillar_pixel_width + right_pillar_pixel_width) / 2
        left_box_width = left_box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        right_box_width = right_box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * PILLAR_WIDTH

        print("横向间隙尺寸\n间隙1:%.2f\n间隙4:%.2f" % (left_pillar_box_gap_width + LEFT_OFFSET, right_pillar_box_gap_width + RIGHT_OFFSET), file=file)
        print("横向货物尺寸\n货物1:%.2f\n货物3:%.2f" % (left_box_width, right_box_width), file=file)
        print("库位中中间位置的货物不存在，其他位置存在货物，库位中部处于安全距离。", file=file)
        close_file_description(file)
        return

    if flag4 and flag5 and flag3 and not flag1:
        left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
        left_box_pixel_width = points5[1][0] - points5[0][0]
        left_pillar_box_gap_pixel_width = points5[0][0] - left_pillar_right_x
        left_box_gap_pixel_width = points4[2][0] - points4[1][0]
        center_box_pixel_width = points3[1][0] - points3[0][0]

        pillar_pixel_width = left_pillar_pixel_width
        center_box_width = center_box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        left_box_width = left_box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        left_box_gap_width = left_box_gap_pixel_width / pillar_pixel_width * PILLAR_WIDTH

        print("横向间隙尺寸\n间隙1:%.2f\n间隙2:%.2f" % (left_pillar_box_gap_width + LEFT_OFFSET, left_box_gap_width + LEFT_CENTER_OFFSET), file=file)
        print("横向货物尺寸\n货物1:%.2f\n货物2:%.2f" % (left_box_width, center_box_width), file=file)
        print("库位中右侧位置的货物不存在，其他位置存在货物，库位右侧处于安全距离。", file=file)
        close_file_description(file)
        return

    if flag2 and flag1 and flag3 and not flag5:
        right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
        right_box_pixel_width = points1[1][0] - points1[0][0]
        right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]
        right_box_gap_pixel_width = points2[2][0] - points2[1][0]
        center_box_pixel_width = points3[1][0] - points3[0][0]

        pillar_pixel_width = right_pillar_pixel_width
        center_box_width = center_box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        right_box_width = right_box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * PILLAR_WIDTH
        right_box_gap_width = right_box_gap_pixel_width / pillar_pixel_width * PILLAR_WIDTH

        print("横向间隙尺寸\n间隙3:%.2f\n间隙4:%.2f" % (right_box_gap_width + RIGHT_CENTER_OFFSET, right_pillar_box_gap_width + RIGHT_OFFSET), file=file)
        print("横向货物尺寸\n货物2:%.2f\n货物3:%.2f" % (center_box_width, right_box_width), file=file)
        print("库位中左侧位置的货物不存在，其他位置存在货物，库位左侧处于安全距离。", file=file)
        close_file_description(file)
        return

    left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
    right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
    center_box_pixel_width = points3[1][0] - points3[0][0]
    left_box_pixel_width = points5[1][0] - points5[0][0]
    right_box_pixel_width = points1[1][0] - points1[0][0]
    left_pillar_box_gap_pixel_width = points5[0][0] - left_pillar_right_x
    right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]
    left_box_gap_pixel_width = points4[2][0] - points4[1][0]
    right_box_gap_pixel_width = points2[2][0] - points2[1][0]

    pillar_pixel_width = (left_pillar_pixel_width + right_pillar_pixel_width) / 2
    center_box_width = center_box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
    left_box_width = left_box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
    right_box_width = right_box_pixel_width / pillar_pixel_width * PILLAR_WIDTH
    left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * PILLAR_WIDTH
    right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * PILLAR_WIDTH
    left_box_gap_width = left_box_gap_pixel_width / pillar_pixel_width * PILLAR_WIDTH
    right_box_gap_width = right_box_gap_pixel_width / pillar_pixel_width * PILLAR_WIDTH

    scale_factor = (2 * PILLAR_WIDTH + center_box_width + left_box_width + right_box_width\
                    + left_pillar_box_gap_width + right_pillar_box_gap_width + left_box_gap_width + right_box_gap_width)\
                        / KUWEI_WIDTH
    print("scale_factor: ", scale_factor)

    center_box_width /= scale_factor
    left_box_width /= scale_factor
    right_box_width /= scale_factor
    left_pillar_box_gap_width /= scale_factor
    right_pillar_box_gap_width /= scale_factor
    left_box_gap_width /= scale_factor
    right_box_gap_width /= scale_factor

    print("横向间隙尺寸\n间隙1:%.2f\n间隙2:%.2f\n间隙3:%.2f\n间隙4:%.2f" % (left_pillar_box_gap_width + LEFT_OFFSET, left_box_gap_width + LEFT_CENTER_OFFSET,\
                                                            right_box_gap_width + RIGHT_CENTER_OFFSET, right_pillar_box_gap_width + RIGHT_OFFSET), file=file)
    print("横向货物尺寸\n货物1:%.2f\n货物2:%.2f\n货物3:%.2f" % (left_box_width, center_box_width, right_box_width), file=file)
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
        print("Measuring " + dir + " horizontal size......")
        try:
            Serial_Images_Measurement(os.path.join(data_src_dir, dir), os.path.join(data_dst_dir, dir))
        except Exception as e:
            f = get_file_description(data_dst_dir, 'fail_log.txt')
            f.write(dir + " horizontal measurement fail! Please check this kuwei.")
            print("Exception info: " + repr(e), file=f)
            close_file_description(f)

    print("Measurement task complete!")


if __name__ == '__main__':
    # Stitching_Util_main(image_dir)
    # Stitching_main(image_dir)
    # Opencv_Stitching_main(image_dir)
    # Compose_Measurement(image_dir)
    # Serial_Images_Measurement(image_dir, save_dir)
    batch_serial_measurement(data_src_dir, data_dst_dir)