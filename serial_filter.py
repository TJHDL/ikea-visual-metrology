# -*- coding: utf-8 -*-
# @Time    : 2023/10/31 13:44
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : serial_filter.py

import os
import cv2
import numpy as np
from utils.semantic_info_util import LEDNet_inference
from utils.marking_points_util import BoxMPR_inference
from utils import save_key_frames, get_head_tail_sorted_number, get_file_description, close_file_description
import parameters as param

image_dir = r'C:\Users\95725\Desktop\rtsp_picture_20240322\floor4'
save_dir = r'C:\Users\95725\Desktop\src'
result_path = r'C:\Users\95725\Desktop\semantic_result'

image_num = 540
total_num = 2000


'''
    从图像候选集中选择出最合适的一帧
'''
def get_perfect_key_frame(fig_candidates, fd):
    point_num_arr = []

    for idx, image in enumerate(fig_candidates):
        # image = cv2.convertScaleAbs(image, alpha=3, beta=0)
        points = BoxMPR_inference(image)
        point_num_arr.append(len(points))

    max_num = np.max(point_num_arr)
    x_arr = np.where(point_num_arr == max_num)[0]

    key_idx = x_arr[int(len(x_arr) / 2)]

    # print("检测到的最大角点数: ", max_num)
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
    # print("Circle position: x=" + str(pillar_x) + " y=" + str(int(height / 2)))
    # cv2.circle(raw_image, (pillar_x, int(height / 2)), radius, (0, 0, 255), 3)
    # cv2.namedWindow(str(serial_num), 0)
    # cv2.imshow(str(serial_num), raw_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return pillar_x


'''
    获取序列图片中的库位开始点和结束点对应的图片序号
'''
def get_kuwei_range(serial_num, total_num, kuwei_type):
    #   state=0: 未找到开始点 state=1: 已找到开始点但未找到结束点 state=2: 已找到结束点
    state = 0
    cnt = 0
    cnt_threshold = param.KUWEI_TYPE_THRESHOLD_CHOICES[kuwei_type]
    start_num = serial_num
    end_num = serial_num
    serial_images = []
    pillar_x = 0
    pre_pillar_x = 0

    while True:
        cnt += 1
        # print("Detecting " + str(serial_num) + ".jpg.....")
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
        # print("pillar x pos: ", x_ratio)

        # 避免库位划分时语义信息误识别导致的突变
        gap_x_ratio = min(abs(x_ratio - pre_x_ratio), min(x_ratio, pre_x_ratio) + 1 - max(x_ratio, pre_x_ratio))
        if cnt > 1 and gap_x_ratio >= 0.1:
            serial_num += 1
            continue

        if state == 0 and (param.START_POS_RANGE_LEFT <= x_ratio and x_ratio <= param.START_POS_RANGE_RIGHT):
            start_num = serial_num
            cnt = 0
            state = 1
        elif state == 1 and cnt >= cnt_threshold and (param.END_POS_RANGE_LEFT <= x_ratio and x_ratio <= param.END_POS_RANGE_RIGHT):
            end_num = serial_num
            cnt = 0
            state = 2

        if state == 1 or state == 2:
            serial_images.append(raw_image)

        serial_num += 1
        pre_pillar_x = pillar_x
        if cnt >= param.MAX_KUWEI_IMAGES_COUNT or state == 2 or serial_num > total_num:
            break

    if state == 0:
        print("[WARNING] 未找到开始点")
        return -1, -1, None
    elif state == 1:
        print("[ERROR] 未找到结束点")
        return -1, -1, None

    return start_num, end_num, serial_images


'''
    关键帧抽取状态机
'''
def key_frame_extractor(serial_images, center_offset, fd, kuwei_type):
    key_frames = []

    if kuwei_type == param.KUWEI_TYPE_3:
        #   state=0: 未找到Fig1 state=1: 已找到Fig1但未找到Fig2 state=2: 已找到Fig2但未找到Fig3
        #   state=3: 已找到Fig3但未找到Fig4 state=4: 已找到Fig4但未找到Fig5 state=5: 已找到Fig5
        state = 0
        fig2_candidates = []
        fig4_candidates = []

        fig2_start_offset = center_offset - param.KUWEI_TYPE_3_FIG2_MAX_OFFSET
        fig2_end_offset = center_offset - param.KUWEI_TYPE_3_FIG2_MIN_OFFSET
        fig4_start_offset = center_offset + param.KUWEI_TYPE_3_FIG4_MIN_OFFSET
        fig4_end_offset = center_offset + param.KUWEI_TYPE_3_FIG4_MAX_OFFSET

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
    elif kuwei_type == param.KUWEI_TYPE_2:
        #   state=0: 未找到Fig1 state=1: 已找到Fig1但未找到Fig2 state=2: 已找到Fig2但未找到Fig3 state=3: 已找到Fig3
        state = 0
        fig2_candidates = []

        fig2_start_offset = center_offset - param.KUWEI_TYPE_2_FIG2_LR_OFFSET
        fig2_end_offset = center_offset + param.KUWEI_TYPE_2_FIG2_LR_OFFSET

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
            elif state == 2 and idx == len(serial_images) - 1:
                key_frames.append(image)
                state = 3

    return key_frames


'''
    实现对单个库位的自动关键帧筛选
'''
def single_kuwei_key_frame_filter(image_num, save_path, total_num, fd, kuwei_type):
    start_num, end_num, serial_images = get_kuwei_range(image_num, total_num, kuwei_type)
    if start_num == -1 or end_num == -1:
        return -1, -1

    kuwei_str = "起始序号: " + str(start_num) + " 终止序号: " + str(end_num) + " 库位图片数量: " + str(len(serial_images))
    print("[INFO] " + kuwei_str)
    fd.write(kuwei_str + "\n")
    fd.flush()
    center_offset = int((end_num - start_num) / 2)
    key_frames = key_frame_extractor(serial_images, center_offset, fd, kuwei_type)
    save_key_frames(key_frames, save_path)

    return start_num, end_num


'''
    批量完成序列化图片筛选
'''
def batch_kuwei_key_frame_filter(image_dir, save_dir):
    print("[WORK FLOW] Starting spliting kuweis.")
    current_num, total_num = get_head_tail_sorted_number(image_dir)
    kuwei_num = 1
    f = get_file_description(save_dir, 'filter_log.txt')
    while current_num <= total_num:
        kuwei_type = param.KUWEI_TYPE_3
        save_path = os.path.join(save_dir, "kuwei" + str(kuwei_num))
        f.write("\nkuwei" + str(kuwei_num) + ":" + "\n")
        f.flush()
        _, end_num = single_kuwei_key_frame_filter(current_num, save_path, total_num, f, kuwei_type)
        kuwei_num += 1
        if end_num == -1:
            current_num += param.KUWEI_TYPE_THRESHOLD_CHOICES[kuwei_type]
            continue
        current_num = end_num + 1
        # print("Save path: " + save_path)

    close_file_description(f)
    print("[INFO] Auto key frame filter complete!")
    print("[INFO] Filtered kuwei num: " + str(len([dir_name for dir_name in os.listdir(save_dir) if not dir_name.endswith(".txt")])))
    print("[WORK FLOW] Spliting kuweis complete.")


if __name__ == '__main__':
    # f = get_file_description(save_dir, 'test_log.txt')
    # single_kuwei_key_frame_filter(image_num, save_dir, total_num, f)
    batch_kuwei_key_frame_filter(image_dir, save_dir)