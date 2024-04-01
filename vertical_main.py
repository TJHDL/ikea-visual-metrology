# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 16:34
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : vertical_main.py
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parameters as param
from utils.semantic_info_util import LEDNet_inference, upside_detect, gap_height_measurement_mask, gap_height_measurement_based_on_camera_height, fixed_error_correction
from utils.line_fit import upside_line_detect
from utils import get_file_description, close_file_description
from utils.marking_points_util import BoxMPR_inference, points_filter
from utils.depth_util import key_area_depth, depth_correction

image_dir = r'C:\Users\95725\Desktop\src\kuwei18'
save_dir = r'C:\Users\95725\Desktop\dst\kuwei18'
data_src_dir = r'C:\Users\95725\Desktop\src'
data_dst_dir = r'C:\Users\95725\Desktop\dst'


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
    BoxMPR主函数入口
'''
def BoxMPR_LEDNet_main(image_dir, save_dir, image_name):
    image = cv2.imread(os.path.join(image_dir, image_name))
    height = image.shape[0]
    width = image.shape[1]
    points = BoxMPR_inference(image)
    if len(points) < 2:
        print("[INFO] Empty place.")
        return -1, False

    point_pairs = points_filter(points, image)

    if (len(point_pairs) == 0):
        print("[INFO] Empty place.")
        return -1, False

    if len(point_pairs[0]) != 2:
        print("[ERROR] " + image_name + " points number error! Points num: %d" % len(point_pairs[0]))
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

    w = int(((p3_x - p2_x) + (p1_x - p0_x)) / 2)
    h = int(((p1_y - p3_y) + (p0_y - p2_y)) / 2)
    
    mask = np.array(mask)
    ROI = parallelogram_to_rectangle(mask, w, h, p0, p1, p2, p3)

    # cv2.namedWindow(image_name, 0)
    # cv2.resizeWindow(image_name, w, h)
    # cv2.imshow(image_name, ROI)
    # cv2.waitKey(0)

    # red_width = 9.8
    # gap_height = gap_height_measurement_mask(ROI, w, h, RED_WIDTH)
    gap_height = gap_height_measurement_based_on_camera_height(ROI, w, h, param.H_CAMERA, [p0_x, p0_y], [p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y])
    print("[INFO] Gap's height: ", gap_height)

    # cv2.imwrite(os.path.join(save_dir, image_name.split('.')[0] + "_" + str(gap_height) + "." + image_name.split('.')[1]), image)
    cv2.imwrite(os.path.join(save_dir, 'vertical_' + image_name), image)
    return gap_height, True, [p0_x, p0_y], [p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]


'''
    KUWEI_TYPE_3型库位测量
'''
def measure_kuwei_type_3(image_dir, save_dir, file):
    gap_height1, flag1 = -1, True
    gap_height3, flag3 = -1, True
    gap_height5, flag5 = -1, True
    try:
        gap_height1, flag1, fig1_point1, fig1_point2, fig1_pillar_point1, fig1_pillar_point2 = BoxMPR_LEDNet_main(image_dir, save_dir, '1.jpg')
        # print("fig1_point1: " + str(fig1_point1))
    except Exception as e:
        # print("Fig1: " + repr(e))
        flag1 = False
    try:
        gap_height3, flag3, fig3_point1, fig3_point2, fig3_pillar_point1, fig3_pillar_point2 = BoxMPR_LEDNet_main(image_dir, save_dir, '3.jpg')
        # print("fig3_point1: " + str(fig3_point1))
    except Exception as e:
        # print("Fig3: " + repr(e))
        flag3 = False
    try:
        gap_height5, flag5, fig5_point1, fig5_point2, fig5_pillar_point1, fig5_pillar_point2 = BoxMPR_LEDNet_main(image_dir, save_dir, '5.jpg')
        # print("fig5_point1: " + str(fig5_point1))
    except Exception as e:
        # print("Fig5: " + repr(e))
        flag5 = False

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
    KUWEI_TYPE_2型库位测量
'''
def measure_kuwei_type_2(image_dir, save_dir, file):
    gap_height1, flag1 = -1, True
    gap_height3, flag3 = -1, True
    try:
        gap_height1, flag1, fig1_point1, fig1_point2, fig1_pillar_point1, fig1_pillar_point2 = BoxMPR_LEDNet_main(image_dir, save_dir, '1.jpg')
        # print("fig1_point1: " + str(fig1_point1))
    except Exception as e:
        # print("Fig1: " + repr(e))
        flag1 = False
    try:
        gap_height3, flag3, fig3_point1, fig3_point2, fig3_pillar_point1, fig3_pillar_point2 = BoxMPR_LEDNet_main(image_dir, save_dir, '3.jpg')
        # print("fig3_point1: " + str(fig3_point1))
    except Exception as e:
        # print("Fig3: " + repr(e))
        flag3 = False

    if flag1 and not flag3:
        print("纵向间隙尺寸\n间隙2:%.2f" % (gap_height1), file=file)
        print("库位中只放置了右侧货物", file=file)
        close_file_description(file)
        return

    if not flag1 and flag3:
        print("纵向间隙尺寸\n间隙1:%.2f" % (gap_height3), file=file)
        print("库位中只放置了左侧货物", file=file)
        close_file_description(file)
        return

    if not flag1 and not flag3:
        print("库位中无货物，均处于安全距离", file=file)
        close_file_description(file)
        return

    print("纵向间隙尺寸\n间隙1:%.2f\n间隙2:%.2f" % (gap_height3, gap_height1), file=file)
    close_file_description(file)

    return


def Serial_Images_Measurement(image_dir, save_dir, kuwei_type):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file = get_file_description(save_dir, "measurement.txt")

    if kuwei_type == param.KUWEI_TYPE_3:
        measure_kuwei_type_3(image_dir, save_dir, file)
    elif kuwei_type == param.KUWEI_TYPE_2:
        measure_kuwei_type_2(image_dir, save_dir, file)

    return 


'''
    批量完成图片的序列化测量
'''
def batch_serial_measurement(data_src_dir, data_dst_dir):
    print("[WORK FLOW] Starting measuring vertical size.")
    dirs = os.listdir(data_src_dir)
    for dir in dirs:
        if dir.endswith('.txt'):
            continue
        print("[INFO] Measuring " + dir + " vertical size......")
        kuwei_type = param.KUWEI_TYPE_3
        try:
            Serial_Images_Measurement(os.path.join(data_src_dir, dir), os.path.join(data_dst_dir, dir), kuwei_type)
        except Exception as e:
            f = get_file_description(data_dst_dir, 'fail_log.txt')
            f.write(dir + " vertical measurement fail! Please check this kuwei.")
            print("Exception info: " + repr(e), file=f)
            close_file_description(f)

    print("[INFO] Measurement task complete!")
    print("[WORK FLOW] Measuring vertical size complete.")


'''
    批量完成图片的序列化测量(protocol)
'''
def batch_serial_measurement_protocol(data_src_dir, data_dst_dir):
    print("[WORK FLOW] Starting measuring vertical size.")
    dirs = os.listdir(data_src_dir)
    for dir in dirs:
        if dir.endswith('.txt'):
            continue
        print("[INFO] Measuring " + dir + " vertical size......")
        kuwei_type = int(dir.split('_')[-1])
        param.FLOOR_NUM = int(dir.split('_')[2])
        param.H_CAMERA = param.FLOOR_NUM * param.FLOOR_HEIGHT \
            - (param.CAR_HEIGHT + param.UAV_HEIGHT[param.FLOOR_NUM]) - param.TIEPIAN_WIDTH
        try:
            Serial_Images_Measurement(os.path.join(data_src_dir, dir), os.path.join(data_dst_dir, dir), kuwei_type)
        except Exception as e:
            f = get_file_description(data_dst_dir, 'fail_log.txt')
            f.write(dir + " vertical measurement fail! Please check this kuwei.")
            print("Exception info: " + repr(e), file=f)
            close_file_description(f)

    print("[INFO] Measurement task complete!")
    print("[WORK FLOW] Measuring vertical size complete.")


if __name__ == '__main__':
    # Serial_Images_Measurement(image_dir, save_dir)
    batch_serial_measurement(data_src_dir, data_dst_dir)
    # batch_serial_measurement_protocol(data_src_dir, data_dst_dir)