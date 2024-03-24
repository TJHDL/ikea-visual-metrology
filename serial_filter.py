# -*- coding: utf-8 -*-
# @Time    : 2023/10/31 13:44
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : serial_filter.py

import os
import cv2
import numpy as np
import config
import torch
from semantic_info import LEDNet_inference
from model import DirectionalPointDetector
from torchvision.transforms import ToTensor
from data import get_predicted_points, calc_point_squre_dist
from utils import save_key_frames, get_head_tail_sorted_number, get_file_description, close_file_description

image_dir = r'C:\Users\95725\Desktop\rtsp_picture_20240322\floor4'
save_dir = r'C:\Users\95725\Desktop\src'
result_path = r'C:\Users\95725\Desktop\semantic_result'
BoxMPR_detector_weights = r'checkpoints\dp_detector_59_dark.pth'   #开灯:r'checkpoints\dp_detector_799_v100.pth' 关灯:r'checkpoints\dp_detector_59_dark.pth'
image_num = 540
total_num = 2000

BOXMPR_DEVICE = None
BOXMPR_MODEL = None

'''
    利用检测器检测单张图片中的一对顶点
'''
def detect_image(detector, device, image):
    pred_points = detect_marking_points(detector, image, 0.05, device)  #threshold:0.05
    return get_cornerPoints(image, pred_points)

'''
    Given image read from opencv, return detected marking points.
'''
def detect_marking_points(detector, image, thresh, device):
    prediction = detector(preprocess_image(image).to(device))
    return get_predicted_points(prediction[0], thresh)

'''
    Preprocess numpy image to torch tensor.
'''
def preprocess_image(image):
    if image.shape[0] != 512 or image.shape[1] != 512:
        image = cv2.resize(image, (512, 512))
    return torch.unsqueeze(ToTensor()(image), 0)

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
    num_detected = len(points)
    point_pairs = []
    BOX_MAX_X_DIST = 480
    BOX_MIN_X_DIST = 220
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
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
            if distance < 60000 or distance > 200000:
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
            if delta_x < config.BOX_MIN_X_DIST or delta_x > BOX_MAX_X_DIST: # config.BOX_MIN_X_DIST
                continue

            # Step 7: bottom points filtration
            if point_i[1] >= 550 or point_j[1] >= 550:
                continue

            point_pairs.append((i, j))

    return point_pairs

'''
    从图像候选集中选择出最合适的一帧
'''
def get_perfect_key_frame(fig_candidates, fd):
    point_num_arr = []

    for idx, image in enumerate(fig_candidates):
        # image = cv2.convertScaleAbs(image, alpha=3, beta=0)
        points = BoxMPR_inference(image)
        point_num_arr.append(len(points))

    # distribution = np.bincount(point_num_arr)
    # max_frequency = np.max(distribution)
    # x_arr = np.where(distribution == max_frequency)[0]
    max_num = np.max(point_num_arr)
    # if max_num >= 4:
    #     x_arr = np.where(point_num_arr == max_num)[0]
    #     four_arr = np.where(point_num_arr == 4)[0]
    #     for element in four_arr:
    #         x_arr.append(element)
    # else:
    #     x_arr = np.where(point_num_arr == max_num)[0]
    x_arr = np.where(point_num_arr == max_num)[0]

    key_idx = x_arr[int(len(x_arr) / 2)]

    print("检测到的最大角点数: ", max_num)
    fd.write("检测到的最大角点数: " + str(max_num) + "\n")
    fd.write("包含以下帧id: \n")
    for idx in x_arr:
        fd.write(str(idx) + ".")
    fd.write("\n被选中的关键帧序号: " + str(key_idx) + "\n")
    fd.flush()

    return fig_candidates[key_idx]

'''
    计算图像中货架立柱的中心位置以确定其在图像中方位
'''
def calculate_pillar_center(predict):
    sample_line = int(predict.shape[0] / 2)
    x_coordinates = np.where(predict[sample_line] == 2)[0]
    if len(x_coordinates) == 0:
        return -1
    distribution = np.bincount(x_coordinates)
    max_frequency = np.max(distribution)
    x_arr = np.where(distribution == max_frequency)[0]
    pillar_x = int(np.mean(x_arr))

    return pillar_x

'''
    根据语义信息定位图片中货架立柱的横向位置
'''
def detect_pillar_position(raw_image, serial_num, width, height):
    RGB_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    predict, mask = LEDNet_inference(RGB_image)
    # mask.save(os.path.join(result_path, ".".join([str(serial_num), 'png'])))

    pillar_x = calculate_pillar_center(predict)
    if pillar_x == -1:
        return pillar_x

    radius = 5
    print("Circle position: x=" + str(pillar_x) + " y=" + str(int(height / 2)))
    # cv2.circle(raw_image, (pillar_x, int(height / 2)), radius, (0, 0, 255), 3)
    # cv2.namedWindow(str(serial_num), 0)
    # cv2.imshow(str(serial_num), raw_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return pillar_x

'''
    获取序列图片中的库位开始点和结束点对应的图片序号
'''
def get_kuwei_range(serial_num, total_num):
    #   state=0: 未找到开始点 state=1: 已找到开始点但未找到结束点 state=2: 已找到结束点
    state = 0
    cnt = 0
    start_num = serial_num
    end_num = serial_num
    start_pos_range_left = 0.686    # 12 / 17
    start_pos_range_right = 0.785   # 13 / 17
    end_pos_range_left = 0.215  # 4 / 17
    end_pos_range_right = 0.314  # 5 / 17
    serial_images = []
    pillar_x = 0
    pre_pillar_x = 0

    while True:
        cnt += 1
        print("Detecting " + str(serial_num) + ".jpg.....")
        try:
            raw_image = cv2.imread(os.path.join(image_dir, str(serial_num) + ".jpg"))
        except Exception as e:
            print(str(serial_num) + ".jpg 图片缺失")
            serial_num += 1
            continue
        if raw_image is None:
            print(str(serial_num) + ".jpg 图片缺失")
            serial_num += 1
            continue
        width = raw_image.shape[1]
        height = raw_image.shape[0]
        pillar_x = detect_pillar_position(raw_image, serial_num, width, height)
        x_ratio = pillar_x / width
        pre_x_ratio = pre_pillar_x / width
        print("pillar x pos: ", x_ratio)

        # 避免库位划分时语义信息误识别导致的突变
        gap_x_ratio = min(abs(x_ratio - pre_x_ratio), min(x_ratio, pre_x_ratio) + 1 - max(x_ratio, pre_x_ratio))
        if cnt > 1 and gap_x_ratio >= 0.1:
            serial_num += 1
            continue

        if state == 0 and (start_pos_range_left <= x_ratio and x_ratio <= start_pos_range_right):
            start_num = serial_num
            state = 1
        elif state == 1 and cnt >= 50 and (end_pos_range_left <= x_ratio and x_ratio <= end_pos_range_right):
            end_num = serial_num
            state = 2

        if state == 1 or state == 2:
            serial_images.append(raw_image)

        serial_num += 1
        pre_pillar_x = pillar_x
        if cnt >= 80 or state == 2 or serial_num > total_num:
            break

    if state == 0:
        print("未找到开始点")
        return -1, -1, None
    elif state == 1:
        print("未找到结束点")
        return -1, -1, None

    return start_num, end_num, serial_images


def key_frame_extractor(serial_images, center_offset, fd):
    #   state=0: 未找到Fig1 state=1: 已找到Fig1但未找到Fig2 state=2: 已找到Fig2但未找到Fig3
    #   state=3: 已找到Fig3但未找到Fig4 state=4: 已找到Fig4但未找到Fig5 state=5: 已找到Fig5
    state = 0
    key_frames = []
    fig2_candidates = []
    fig4_candidates = []
    # center_offset = 25
    fig2_max_offset, fig2_min_offset = 16, 6  # 16, 6
    fig4_max_offset, fig4_min_offset = 18, 8  # 16, 6
    fig2_start_offset = center_offset - fig2_max_offset
    fig2_end_offset = center_offset - fig2_min_offset
    fig4_start_offset = center_offset + fig4_min_offset
    fig4_end_offset = center_offset + fig4_max_offset

    for idx, image in enumerate(serial_images):
        if state == 0:
            key_frames.append(image)
            state = 1
        elif state == 1:
            if fig2_start_offset <= idx and idx <= fig2_end_offset:
                fig2_candidates.append(image)
                continue
            if idx > fig2_end_offset:
                key_frames.append(get_perfect_key_frame(fig2_candidates, fd))
                state = 2
        elif state == 2 and idx == center_offset:
            key_frames.append(image)
            state = 3
        elif state == 3:
            if fig4_start_offset <= idx and idx <= fig4_end_offset:
                fig4_candidates.append(image)
                continue
            if idx > fig4_end_offset:
                key_frames.append(get_perfect_key_frame(fig4_candidates, fd))
                state = 4
        elif state == 4 and idx == len(serial_images) - 1:
            key_frames.append(image)
            state = 5

    return key_frames

'''
    实现对单个库位的自动关键帧筛选
'''
def single_kuwei_key_frame_filter(image_num, save_path, total_num, fd):
    start_num, end_num, serial_images = get_kuwei_range(image_num, total_num)
    if start_num == -1 or end_num == -1:
        return -1, -1

    kuwei_str = "起始序号: " + str(start_num) + " 终止序号: " + str(end_num) + " 库位图片数量: " + str(len(serial_images))
    print(kuwei_str)
    fd.write(kuwei_str + "\n")
    fd.flush()
    center_offset = int((end_num - start_num) / 2)
    key_frames = key_frame_extractor(serial_images, center_offset, fd)
    save_key_frames(key_frames, save_path)

    return start_num, end_num

'''
    批量完成序列化图片筛选
'''
def batch_kuwei_key_frame_filter(image_dir, save_dir):
    current_num, total_num = get_head_tail_sorted_number(image_dir)
    kuwei_num = 1
    f = get_file_description(save_dir, 'filter_log.txt')
    while current_num <= total_num:
        save_path = os.path.join(save_dir, "kuwei" + str(kuwei_num))
        f.write("\nkuwei" + str(kuwei_num) + ":" + "\n")
        f.flush()
        _, end_num = single_kuwei_key_frame_filter(current_num, save_path, total_num, f)
        kuwei_num += 1
        if end_num == -1:
            current_num += 50
            continue
        current_num = end_num + 1
        print("Save path: " + save_path)

    close_file_description(f)
    print("Auto key frame filter complete!")
    print("Filtered kuwei num: " + str(len([dir_name for dir_name in os.listdir(save_dir) if not dir_name.endswith(".txt")])))


if __name__ == '__main__':
    # f = get_file_description(save_dir, 'test_log.txt')
    # single_kuwei_key_frame_filter(image_num, save_dir, total_num, f)
    batch_kuwei_key_frame_filter(image_dir, save_dir)