# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 17:52
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : semantic_info.py
import os
import cv2
import torch
from torchvision import transforms
from model.lednet import LEDNet
import utils as ptutil
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_dir = r'D:\ProjectCodes\VisionMeasurement\test'
image_name = r'IMG_3543.jpg'
result_path = r'C:\Users\95725\Desktop\semantic_result'
save_dir = r'D:\ProjectCodes\VisionMeasurement\result'
LEDNet_detector_weights = r'checkpoints\LEDNet_iter_170400_v100.pth'

LEDNET_DEVICE = None
LEDNET_MODEL = None
TRANSFORM_FUNCTION = None

'''
    通过状态机的形式判断此时像素点位于立柱外、立柱中、立柱内
    predict[x, y] = 0-其他 1-货物 2-立柱
    direction = 0 -> 左边立柱
    direction = 1 -> 右边立柱
'''
def pillar_point_check(predict, threshold, x, y, width, direction):
    left_x = 0
    right_x = 0
    if direction == 0:
        for i in range(threshold, x):
            flag = True
            for offset in range(1, threshold):
                if predict[y, i - offset] == 2 or predict[y, i + offset] != 2:
                    flag = False
                    break
            if flag:
                left_x = i
        for i in range(x - threshold, 0, -1):
            flag = True
            for offset in range(1, threshold):
                if predict[y, i - offset] != 2 or predict[y, i + offset] == 2:
                    flag = False
                    break
            if flag:
                right_x = i
    elif direction == 1:
        for i in range(width - threshold - 1, x, -1):
            flag = True
            for offset in range(1, threshold):
                if predict[y, i - offset] != 2 or predict[y, i + offset] == 2:
                    flag = False
                    break
            if flag:
                right_x = i
        for i in range(x + threshold, width - 1):
            flag = True
            for offset in range(1, threshold):
                if predict[y, i - offset] == 2 or predict[y, i + offset] != 2:
                    flag = False
                    break
            if flag:
                left_x = i

    return left_x, right_x

'''
    检测立柱的左右边界位置
'''
def pillar_detect(predict, left_x, right_x, y, width):
    # 根据LEDNet对像素类别预测结果的变化情况判断边沿点
    # 状态机：0->库位立柱以外 1->立柱 2->库位立柱以内
    left_pillar_left_x = 0
    left_pillar_left_y = y + 10
    left_pillar_right_x = 0
    left_pillar_right_y = y + 10

    right_pillar_left_x = 0
    right_pillar_left_y = y + 10
    right_pillar_right_x = 0
    right_pillar_right_y = y + 10

    # 左顶点
    left_pillar_left_x, left_pillar_right_x = pillar_point_check(predict, 20, left_x, left_pillar_left_y, width, 0)  # 25

    # 右顶点
    right_pillar_left_x, right_pillar_right_x = pillar_point_check(predict, 20, right_x, right_pillar_right_y, width, 1)  # 25

    return left_pillar_left_x, left_pillar_right_x, right_pillar_left_x, right_pillar_right_x

'''
    检测立柱的左右边界位置(不拼接图片，只识别单张图片中的两对角点)
'''
def pillar_detect_partial(predict, x, y, width, direction):

    pillar_left_x, pillar_right_x = pillar_point_check(predict, 10, x, y, width, direction)   #10

    return pillar_left_x, pillar_right_x

'''
    通过状态机的形式判断此时像素点位于横梁下、横梁中、横梁上
    predict[x, y] = 0-其他 1-货物 2-横梁
'''
def semantic_point_check(predict, threshold1, threshold2, p_x, p_y):
    state = 0

    for y in range(p_y - threshold1, threshold1, -1):
        if state == 0:
            flag = True

            for i in range(y + threshold1, y, -1):
                if predict[i, p_x] != 0 and predict[i, p_x] != 1:
                    flag = False
                    break
            if not flag:
                continue

            for i in range(y, y - threshold1, -1):
                if predict[i, p_x] != 2:
                    flag = False
                    break
            if not flag:
                continue

            state = 1
        elif state == 1:
            flag = True

            for i in range(y + threshold2, y, -1):
                if predict[i, p_x] != 2:
                    flag = False
                    break
            if not flag:
                continue

            for i in range(y - 1, y - threshold2, -1):
                if predict[i, p_x] != 0 and predict[i, p_x] != 1:
                    flag = False
                    break
            if not flag:
                continue

            state = 2
            return y

    return 0

'''
    检测位于横梁上边沿的像素点坐标
'''
def upside_detect(predict, p0_x, p0_y, p1_x, p1_y):
    # 根据LEDNet对像素类别预测结果的变化情况判断边沿点
    # 状态机：0->横梁下边沿以下 1->横梁 2->横梁上边沿以上
    p2_x = p0_x
    p2_y = 0
    p3_x = p1_x
    p3_y = 0

    # 左顶点
    p2_y = semantic_point_check(predict, 20, 10, p0_x, p0_y)    #20,10

    # 右顶点
    p3_y = semantic_point_check(predict, 20, 10, p1_x, p1_y)    #20,10

    return p2_x, p2_y, p3_x, p3_y

'''
    获取所需的设备及LEDNet模型
'''
def get_device_and_LEDNet_model():
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    model = LEDNet(3).to(device)  # param: num_class
    model.load_state_dict(torch.load(LEDNet_detector_weights, map_location=device))
    model.eval()

    return device, model

'''
    获取所需的转换函数
'''
def get_transform_fn():
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform_fn

'''
    LEDNet语义分割推理
'''
def LEDNet_inference(image):
    global LEDNET_DEVICE, LEDNET_MODEL, TRANSFORM_FUNCTION
    # Load Model
    # device, model = get_device_and_LEDNet_model()
    if LEDNET_DEVICE is None or LEDNET_MODEL is None:
        LEDNET_DEVICE, LEDNET_MODEL = get_device_and_LEDNet_model()

    # Transform
    # transform_fn = get_transform_fn()
    if TRANSFORM_FUNCTION is None:
        TRANSFORM_FUNCTION = get_transform_fn()

    image = TRANSFORM_FUNCTION(image).unsqueeze(0).to(LEDNET_DEVICE)
    # start_time = time.time()
    # f = open(os.path.join(result_path, r'time_log.txt'), 'a')
    with torch.no_grad():
        output = LEDNET_MODEL(image)
    # end_time = time.time()  # 记录结束时间
    # execution_time = end_time - start_time  # 计算执行时间
    # print("推理时间: ", execution_time)
    # f.write(str(execution_time) + "\n")
    # f.flush()
    # f.close()

    predict = torch.argmax(output, 1).squeeze(0).cpu().numpy()
    # print(predict)
    # print(type(predict))

    # 保存并可视化推理结果
    mask = ptutil.get_color_pallete(predict, 'ikea')
    # mask.save(os.path.join(result_path, image_name.replace('jpg', 'png')))
    # mmask = mpimg.imread(os.path.join(result_path, image_name.replace('jpg', 'png')))
    # plt.imshow(mask)
    # plt.show()

    return predict, mask

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
    通过已知的横梁厚度基于语义掩模图像算出横梁与货物之间间隙的高度
'''
def gap_height_measurement_mask(mask, width, height, red_width):
    # # 对二值化图像进行腐蚀操作去除异常边界的干扰
    # erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # img_erosion = cv2.erode(image.copy(), erosion_kernel)

    total_gap = 0
    total_red = 0
    valid_width = 0

    for i in range(width):
        red = 0
        unred = 0
        flag = True
        last_red = 0
        first_red = 0
        for j in range(height):
            if mask[j, i] == 2:
                #   解决横梁区域语义信息不连续的问题
                if last_red != 0 and last_red != j - 1:
                    red += j - last_red - 1
                if first_red == 0:
                    first_red = j
                last_red = j
                red += 1
                unred = 0
            else:
                unred += 1

            #   已穿过横梁区域到达空隙区域，提前退出循环节约时间
            if unred >= int(height / 10):   #threshold=10
                if red == 0:
                    flag = False
                break
            # if unred >= 10:
            #     break

        #   只计算有效的列
        if flag:
            valid_width += 1
            total_red += red
            total_gap += height - red - first_red

    avg_gap = total_gap / valid_width
    avg_red = total_red / valid_width

    # avg_red = height - avg_gap
    print('avg_red: ', avg_red)
    print('avg_gap: ', avg_gap)
    gap_width = avg_gap * (red_width / avg_red)

    # scale_factor = 28 / 18
    scale_factor = 1

    return gap_width * scale_factor

'''
    经验值固定偏差矫正
'''
def fixed_error_correction(gap_width):
    if gap_width >= 60:
        gap_width -= 10
    elif gap_width >= 50:
        gap_width -= 5
    elif gap_width >= 40:
        gap_width -= 4
    elif gap_width >= 33:
        gap_width -= 3
    elif gap_width >= 30:
        gap_width -= 2

    return gap_width

'''
    基于相机相对所谓的'地面'的高度进行目标物尺寸的测量
'''
def gap_height_measurement_based_on_camera_height(mask, width, height, h_camera, point1, point2, point3, point4):
    total_gap = 0
    total_red = 0
    valid_width = 0

    for i in range(width):
        red = 0
        unred = 0
        flag = True
        last_red = 0
        first_red = 0
        for j in range(height):
            if mask[j, i] == 2:
                #   解决横梁区域语义信息不连续的问题
                if last_red != 0 and last_red != j - 1:
                    red += j - last_red - 1
                if first_red == 0:
                    first_red = j
                last_red = j
                red += 1
                unred = 0
            else:
                unred += 1

            #   已穿过横梁区域到达空隙区域，提前退出循环节约时间
            if unred >= int(height / 10):  # threshold=10
                if red == 0:
                    flag = False
                break
            # if unred >= 10:
            #     break

        #   只计算有效的列
        if flag:
            valid_width += 1
            total_red += red
            total_gap += height - red - first_red

    avg_gap = total_gap / valid_width
    avg_red = total_red / valid_width

    # avg_red = height - avg_gap
    print('avg_red: ', avg_red)
    print('avg_gap: ', avg_gap)

    vt = (point1[1] + point2[1]) / 2
    vb = vt - avg_gap
    vo = 360    # 此处的图像用的1088×720，取图像的高度中心即可
    gap_width = h_camera * ((vt - vb) / (vo - vb))

    #   固定偏差矫正（经验值）
    gap_width = fixed_error_correction(gap_width)

    return gap_width


if __name__ == '__main__':
    # image = Image.open(os.path.join(image_dir, image_name))
    image = cv2.imread(os.path.join(image_dir, image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))

    predict, mask = LEDNet_inference(image)

    radius = 5
    p0_x, p0_y = 161, 111
    p1_x, p1_y = 367, 106
    p2_x, p2_y, p3_x, p3_y = upside_detect(predict, p0_x, p0_y, p1_x, p1_y)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.circle(image, (p0_x, p0_y), radius, (255, 0, 0), 3)
    cv2.circle(image, (p1_x, p1_y), radius, (0, 255, 0), 3)
    cv2.circle(image, (p2_x, p2_y), radius, (255, 255, 0), 3)
    cv2.circle(image, (p3_x, p3_y), radius, (0, 255, 255), 3)

    p0 = [p0_x, p0_y]  # 左下角
    p1 = [p1_x, p1_y]  # 右下角
    p2 = [p2_x, p2_y]  # 左上角
    p3 = [p3_x, p3_y]  # 右上角
    w = 160
    h = 240

    mask = np.array(mask)
    ROI = parallelogram_to_rectangle(mask, w, h, p0, p1, p2, p3)
    # plt.imshow(ROI)
    # plt.show()

    red_width = 20
    gap_height = gap_height_measurement_mask(ROI, w, h, red_width)
    print(gap_height)

    # cv2.imwrite(os.path.join(save_dir, "lednet" + image_name), image)
    cv2.namedWindow(image_name, 0)
    cv2.resizeWindow(image_name, 512, 512)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)