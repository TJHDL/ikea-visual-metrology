# -*- coding: utf-8 -*-
# @Time    : 2023/8/14 21:36
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : horizontal_main.py
import os
import cv2
import parameters as param
from utils.semantic_info_util import LEDNet_inference, pillar_detect_partial
from utils import get_file_description, close_file_description
from utils.marking_points_util import BoxMPR_inference, points_filter

image_dir = r'C:\Users\95725\Desktop\src\kuwei6'
save_dir = r'C:\Users\95725\Desktop\dst\kuwei6'
data_src_dir = r'C:\Users\95725\Desktop\src'
data_dst_dir = r'C:\Users\95725\Desktop\dst'


'''
    对标记点进行识别并将数据结构调整为方便处理的形式
'''
def points_extractor(image_dir, img_name, point_num):
    img = cv2.imread(os.path.join(image_dir, img_name))
    points = BoxMPR_inference(img)
    if (point_num == 2 and len(points) < 2) or (point_num == 4 and len(points) < 3):
        # print("length of points: ", len(points))
        # print("此处无货物")
        return points, img, False
    points.sort(key=lambda x: x[0], reverse=False)

    # for point in points:
    #     cv2.circle(img, (point[0], point[1]), 5, (0, 255, 0), 3)

    if point_num == 2:
        point_pairs = points_filter(points, img)
        if(len(point_pairs) == 0):
            # print("此处无货物")
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
        point_pairs = points_filter(points, img)

        if len(point_pairs) != 2:
            print(img_name + " points number error! Points num: %d" % len(points))
            print(img_name + " points number error! Point pairs num: %d" % len(point_pairs))

            if len(points) == 3:
                center_point_cnt = 0
                img_width = img.shape[1]
                for point in points:
                    if point[0] / img_width >= 0.35 and point[0] / img_width <= 0.65:
                        center_point_cnt += 1
            if center_point_cnt == 2:
                return points, img, True

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


def robust_points_gap_location(points, img):
    if len(points) == 3:
        center_point1, center_point2 = None, None
        img_width = img.shape[1]
        for point in points:
            if point[0] / img_width >= param.CENTER_POINT_LEFT_THRESHOLD and point[0] / img_width <= param.CENTER_POINT_RIGHT_THRESHOLD:
                if center_point1 is None:
                    center_point1 = point
                else:
                    center_point2 = point
        box_gap_pixel_width = center_point2[0] - center_point1[0]
    else:
        box_gap_pixel_width = points[2][0] - points[1][0]

    return box_gap_pixel_width

'''
    KUWEI_TYPE_3型库位测量
'''
def measure_kuwei_type_3(image_dir, save_dir, file):
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

    if flag2:
        for point in points2:
            cv2.circle(img2, (point[0], point[1]), radius, (0, 255, 0), 3)

    cv2.imwrite(os.path.join(save_dir, "img2.jpg"), img2)

    if flag3:
        for point in points3:
            cv2.circle(img3, (point[0], point[1]), radius, (0, 255, 0), 3)

    cv2.imwrite(os.path.join(save_dir, "img3.jpg"), img3)

    if flag4:
        for point in points4:
            cv2.circle(img4, (point[0], point[1]), radius, (0, 255, 0), 3)

    cv2.imwrite(os.path.join(save_dir, "img4.jpg"), img4)

    y_line_5 = 0
    left_x_5, right_x_5 = 1024, 0
    if flag5:
        for point in points5:
            y_line_5 = max(y_line_5, int(point[1]))
            left_x_5 = min(left_x_5, int(point[0]))
            right_x_5 = max(right_x_5, int(point[0]))
            cv2.circle(img5, (point[0], point[1]), radius, (0, 255, 0), 3)

    # 左半部分测量
    if flag5:
        left_pillar_left_x, left_pillar_right_x = pillar_detect_partial(predict5, left_x_5 + 20, int(img5.shape[0] / 2), img5.shape[1], 0)
        cv2.circle(img5, (left_pillar_left_x, y_line_5), radius, (0, 0, 255), 3)
        cv2.circle(img5, (left_pillar_right_x, y_line_5), radius, (0, 0, 255), 3)

    cv2.imwrite(os.path.join(save_dir, "img5.jpg"), img5)

    # 右半部分测量
    if flag1:
        right_pillar_left_x, right_pillar_right_x = pillar_detect_partial(predict1, right_x_1 - 20, int(img1.shape[0] / 2), img1.shape[1], 1)
        cv2.circle(img1, (right_pillar_left_x, y_line_1), radius, (0, 0, 255), 3)
        cv2.circle(img1, (right_pillar_right_x, y_line_1), radius, (0, 0, 255), 3)

    cv2.imwrite(os.path.join(save_dir, "img1.jpg"), img1)

    if flag3 and not flag1 and not flag5:
        print("库位中只存在中间位置的货物，两侧不存在货物，处于安全距离。", file=file)
        close_file_description(file)
        return

    if flag5 and not flag3 and not flag1:
        left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
        left_box_pixel_width = points5[1][0] - points5[0][0]
        left_pillar_box_gap_pixel_width = points5[0][0] - left_pillar_right_x
        pillar_pixel_width = left_pillar_pixel_width
        left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        print("横向间隙尺寸\n间隙1:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET), file=file)
        print("横向货物尺寸\n货物1:%.2f" % (left_box_width), file=file)
        print("库位中只存在左侧位置的货物，其他位置不存在货物，右侧处于安全距离。", file=file)
        close_file_description(file)
        return

    if flag1 and not flag3 and not flag5:
        right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
        right_box_pixel_width = points1[1][0] - points1[0][0]
        right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]
        pillar_pixel_width = right_pillar_pixel_width
        right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        print("横向间隙尺寸\n间隙4:%.2f" % (right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
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
        left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH

        print("横向间隙尺寸\n间隙1:%.2f\n间隙4:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET, right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
        print("横向货物尺寸\n货物1:%.2f\n货物3:%.2f" % (left_box_width, right_box_width), file=file)
        print("库位中中间位置的货物不存在，其他位置存在货物，库位中部处于安全距离。", file=file)
        close_file_description(file)
        return

    if flag4 and flag5 and flag3 and not flag1:
        left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
        left_box_pixel_width = points5[1][0] - points5[0][0]
        left_pillar_box_gap_pixel_width = points5[0][0] - left_pillar_right_x
        left_box_gap_pixel_width = robust_points_gap_location(points4, img4)
        center_box_pixel_width = points3[1][0] - points3[0][0]

        pillar_pixel_width = left_pillar_pixel_width
        center_box_width = center_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_box_gap_width = left_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH

        print("横向间隙尺寸\n间隙1:%.2f\n间隙2:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET, left_box_gap_width + param.LEFT_CENTER_OFFSET), file=file)
        print("横向货物尺寸\n货物1:%.2f\n货物2:%.2f" % (left_box_width, center_box_width), file=file)
        print("库位中右侧位置的货物不存在，其他位置存在货物，库位右侧处于安全距离。", file=file)
        close_file_description(file)
        return

    if flag2 and flag1 and flag3 and not flag5:
        right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
        right_box_pixel_width = points1[1][0] - points1[0][0]
        right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]
        right_box_gap_pixel_width = robust_points_gap_location(points2, img2)
        center_box_pixel_width = points3[1][0] - points3[0][0]

        pillar_pixel_width = right_pillar_pixel_width
        center_box_width = center_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_box_gap_width = right_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH

        print("横向间隙尺寸\n间隙3:%.2f\n间隙4:%.2f" % (right_box_gap_width + param.RIGHT_CENTER_OFFSET, right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
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

    left_box_gap_pixel_width = robust_points_gap_location(points4, img4)
    right_box_gap_pixel_width = robust_points_gap_location(points2, img2)

    pillar_pixel_width = (left_pillar_pixel_width + right_pillar_pixel_width) / 2
    center_box_width = center_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    left_box_gap_width = left_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    right_box_gap_width = right_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH

    scale_factor = (2 * param.PILLAR_WIDTH + center_box_width + left_box_width + right_box_width\
                    + left_pillar_box_gap_width + right_pillar_box_gap_width + left_box_gap_width + right_box_gap_width)\
                        / param.KUWEI_WIDTH[param.KUWEI_TYPE_3]
    # print("scale_factor: ", scale_factor)

    center_box_width /= scale_factor
    left_box_width /= scale_factor
    right_box_width /= scale_factor
    left_pillar_box_gap_width /= scale_factor
    right_pillar_box_gap_width /= scale_factor
    left_box_gap_width /= scale_factor
    right_box_gap_width /= scale_factor

    print("横向间隙尺寸\n间隙1:%.2f\n间隙2:%.2f\n间隙3:%.2f\n间隙4:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET, left_box_gap_width + param.LEFT_CENTER_OFFSET,\
                                                            right_box_gap_width + param.RIGHT_CENTER_OFFSET, right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
    print("横向货物尺寸\n货物1:%.2f\n货物2:%.2f\n货物3:%.2f" % (left_box_width, center_box_width, right_box_width), file=file)
    close_file_description(file)

    return


'''
    KUWEI_TYPE_2型库位测量
'''
def measure_kuwei_type_2(image_dir, save_dir, file):
    points1, img1, flag1 = points_extractor(image_dir, '1.jpg', 2)
    points2, img2, flag2 = points_extractor(image_dir, '2.jpg', 4)
    points3, img3, flag3 = points_extractor(image_dir, '3.jpg', 2)

    if not flag1 and not flag2 and not flag3:
        print("库位中无货物，均处于安全距离", file=file)
        close_file_description(file)
        return

    if flag1:
        RGB_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        predict1, mask1 = LEDNet_inference(RGB_image1)

    if flag3:
        RGB_image3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        predict3, mask3 = LEDNet_inference(RGB_image3)

    radius = 5

    y_line_1 = 0
    left_x_1, right_x_1 = 1024, 0
    if flag1:
        for point in points1:
            y_line_1 = max(y_line_1, int(point[1]))
            left_x_1 = min(left_x_1, int(point[0]))
            right_x_1 = max(right_x_1, int(point[0]))
            cv2.circle(img1, (point[0], point[1]), radius, (0, 255, 0), 3)

    if flag2:
        for point in points2:
            cv2.circle(img2, (point[0], point[1]), radius, (0, 255, 0), 3)

    cv2.imwrite(os.path.join(save_dir, "img2.jpg"), img2)

    y_line_3 = 0
    left_x_3, right_x_3 = 1024, 0
    if flag3:
        for point in points3:
            y_line_3 = max(y_line_3, int(point[1]))
            left_x_3 = min(left_x_3, int(point[0]))
            right_x_3 = max(right_x_3, int(point[0]))
            cv2.circle(img3, (point[0], point[1]), radius, (0, 255, 0), 3)

    # 左半部分测量
    if flag3:
        left_pillar_left_x, left_pillar_right_x = pillar_detect_partial(predict3, left_x_3 + 20, int(img3.shape[0] / 2), img3.shape[1], 0)
        cv2.circle(img3, (left_pillar_left_x, y_line_3), radius, (0, 0, 255), 3)
        cv2.circle(img3, (left_pillar_right_x, y_line_3), radius, (0, 0, 255), 3)

    cv2.imwrite(os.path.join(save_dir, "img3.jpg"), img3)

    # 右半部分测量
    if flag1:
        right_pillar_left_x, right_pillar_right_x = pillar_detect_partial(predict1, right_x_1 - 20, int(img1.shape[0] / 2), img1.shape[1], 1)
        cv2.circle(img1, (right_pillar_left_x, y_line_1), radius, (0, 0, 255), 3)
        cv2.circle(img1, (right_pillar_right_x, y_line_1), radius, (0, 0, 255), 3)

    cv2.imwrite(os.path.join(save_dir, "img1.jpg"), img1)

    if flag3 and not flag1:
        left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
        left_box_pixel_width = points3[1][0] - points3[0][0]
        left_pillar_box_gap_pixel_width = points3[0][0] - left_pillar_right_x
        pillar_pixel_width = left_pillar_pixel_width
        left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        print("横向间隙尺寸\n间隙1:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET), file=file)
        print("横向货物尺寸\n货物1:%.2f" % (left_box_width), file=file)
        print("库位中只存在左侧位置的货物，右侧位置不存在货物，右侧处于安全距离。", file=file)
        close_file_description(file)
        return

    if flag1 and not flag3:
        right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
        right_box_pixel_width = points1[1][0] - points1[0][0]
        right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]
        pillar_pixel_width = right_pillar_pixel_width
        right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        print("横向间隙尺寸\n间隙3:%.2f" % (right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
        print("横向货物尺寸\n货物2:%.2f" % (right_box_width), file=file)
        print("库位中只存在右侧位置的货物，左侧位置不存在货物，左侧处于安全距离。", file=file)
        close_file_description(file)
        return

    left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
    right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
    left_box_pixel_width = points3[1][0] - points3[0][0]
    right_box_pixel_width = points1[1][0] - points1[0][0]
    left_pillar_box_gap_pixel_width = points3[0][0] - left_pillar_right_x
    right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]

    center_gap_pixel_width = robust_points_gap_location(points2, img2)

    pillar_pixel_width = (left_pillar_pixel_width + right_pillar_pixel_width) / 2
    left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    center_box_gap_width = center_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH

    scale_factor = (2 * param.PILLAR_WIDTH + center_box_gap_width + left_box_width + right_box_width\
                    + left_pillar_box_gap_width + right_pillar_box_gap_width)\
                        / param.KUWEI_WIDTH[param.KUWEI_TYPE_2]
    # print("scale_factor: ", scale_factor)

    left_box_width /= scale_factor
    right_box_width /= scale_factor
    left_pillar_box_gap_width /= scale_factor
    right_pillar_box_gap_width /= scale_factor
    center_box_gap_width /= scale_factor

    print("横向间隙尺寸\n间隙1:%.2f\n间隙2:%.2f\n间隙3:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET, center_box_gap_width,\
                                                    right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
    print("横向货物尺寸\n货物1:%.2f\n货物2:%.2f" % (left_box_width, right_box_width), file=file)
    close_file_description(file)

    return


'''
    KUWEI_TYPE_4型库位测量
'''
def measure_kuwei_type_4(image_dir, save_dir, file):
    points1, img1, flag1 = points_extractor(image_dir, '1.jpg', 2)
    points2, img2, flag2 = points_extractor(image_dir, '2.jpg', 4)
    points3, img3, flag3 = points_extractor(image_dir, '4.jpg', 4)
    points4, img4, flag4 = points_extractor(image_dir, '6.jpg', 4)
    points5, img5, flag5 = points_extractor(image_dir, '7.jpg', 2)

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

    if flag2:
        for point in points2:
            cv2.circle(img2, (point[0], point[1]), radius, (0, 255, 0), 3)

    cv2.imwrite(os.path.join(save_dir, "img2.jpg"), img2)

    if flag3:
        for point in points3:
            cv2.circle(img3, (point[0], point[1]), radius, (0, 255, 0), 3)

    cv2.imwrite(os.path.join(save_dir, "img3.jpg"), img3)

    if flag4:
        for point in points4:
            cv2.circle(img4, (point[0], point[1]), radius, (0, 255, 0), 3)

    cv2.imwrite(os.path.join(save_dir, "img4.jpg"), img4)

    y_line_5 = 0
    left_x_5, right_x_5 = 1024, 0
    if flag5:
        for point in points5:
            y_line_5 = max(y_line_5, int(point[1]))
            left_x_5 = min(left_x_5, int(point[0]))
            right_x_5 = max(right_x_5, int(point[0]))
            cv2.circle(img5, (point[0], point[1]), radius, (0, 255, 0), 3)

    # 左半部分测量
    if flag5:
        left_pillar_left_x, left_pillar_right_x = pillar_detect_partial(predict5, left_x_5 + 20, int(img5.shape[0] / 2), img5.shape[1], 0)
        cv2.circle(img5, (left_pillar_left_x, y_line_5), radius, (0, 0, 255), 3)
        cv2.circle(img5, (left_pillar_right_x, y_line_5), radius, (0, 0, 255), 3)

    cv2.imwrite(os.path.join(save_dir, "img5.jpg"), img5)

    # 右半部分测量
    if flag1:
        right_pillar_left_x, right_pillar_right_x = pillar_detect_partial(predict1, right_x_1 - 20, int(img1.shape[0] / 2), img1.shape[1], 1)
        cv2.circle(img1, (right_pillar_left_x, y_line_1), radius, (0, 0, 255), 3)
        cv2.circle(img1, (right_pillar_right_x, y_line_1), radius, (0, 0, 255), 3)

    cv2.imwrite(os.path.join(save_dir, "img1.jpg"), img1)

    if flag3 and not flag1 and not flag5:
        print("库位中只存在中间位置的货物，两侧不存在货物，处于安全距离。", file=file)
        close_file_description(file)
        return

    if flag5 and not flag4 and not flag3 and not flag2 and not flag1:
        left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
        left_box_pixel_width = points5[1][0] - points5[0][0]
        left_pillar_box_gap_pixel_width = points5[0][0] - left_pillar_right_x
        pillar_pixel_width = left_pillar_pixel_width
        left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        print("横向间隙尺寸\n间隙1:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET), file=file)
        print("横向货物尺寸\n货物1:%.2f" % (left_box_width), file=file)
        print("库位中只存在左侧位置的货物，其他位置不存在货物，右侧处于安全距离。", file=file)
        close_file_description(file)
        return
    
    if not flag3 and not flag1 and not flag5:
        print("库位中只存在中间位置的货物或是不存在货物，两侧不存在货物，处于安全距离。", file=file)
        close_file_description(file)
        return

    if flag1 and not flag3 and not flag5:
        right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
        right_box_pixel_width = points1[1][0] - points1[0][0]
        right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]
        pillar_pixel_width = right_pillar_pixel_width
        right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        print("横向间隙尺寸\n间隙5:%.2f" % (right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
        print("横向货物尺寸\n货物4:%.2f" % (right_box_width), file=file)
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
        left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH

        print("横向间隙尺寸\n间隙1:%.2f\n间隙5:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET, right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
        print("横向货物尺寸\n货物1:%.2f\n货物4:%.2f" % (left_box_width, right_box_width), file=file)
        print("库位中中间位置的货物不存在，其他位置存在货物，库位中部处于安全距离。", file=file)
        close_file_description(file)
        return
    
    if flag5 and flag4 and not flag3 and not flag2 and not flag1:
        left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
        left_box_pixel_width = points5[1][0] - points5[0][0]
        left_pillar_box_gap_pixel_width = points5[0][0] - left_pillar_right_x
        left_box_gap_pixel_width = robust_points_gap_location(points4, img4)
        
        pillar_pixel_width = left_pillar_pixel_width
        left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_box_gap_width = left_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH

        print("横向间隙尺寸\n间隙1:%.2f\n间隙2:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET, left_box_gap_width + param.LEFT_CENTER_OFFSET), file=file)
        print("横向货物尺寸\n货物1:%.2f" % (left_box_width), file=file)
        print("库位中右半边位置的货物不存在，其他位置存在货物，库位右侧处于安全距离。", file=file)
        close_file_description(file)
        return
    
    if flag1 and flag2 and not flag3 and not flag4 and not flag5:
        right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
        right_box_pixel_width = points1[1][0] - points1[0][0]
        right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]
        right_box_gap_pixel_width = robust_points_gap_location(points2, img2)

        pillar_pixel_width = right_pillar_pixel_width
        right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_box_gap_width = right_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH

        print("横向间隙尺寸\n间隙4:%.2f\n间隙5" % (right_box_gap_width + param.LEFT_CENTER_OFFSET, right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
        print("横向货物尺寸\n货物4:%.2f" % (right_box_width), file=file)
        print("库位中左半边位置的货物不存在，其他位置存在货物，库位左侧处于安全距离。", file=file)
        close_file_description(file)
        return


    if flag5 and flag4 and flag3 and not flag2 and not flag1:
        left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
        left_box_pixel_width = points5[1][0] - points5[0][0]
        left_pillar_box_gap_pixel_width = points5[0][0] - left_pillar_right_x
        left_box_gap_pixel_width = robust_points_gap_location(points4, img4)
        center_box_gap_pixel_width = robust_points_gap_location(points3, img3)

        pillar_pixel_width = left_pillar_pixel_width
        left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_box_gap_width = left_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        center_box_gap_width = center_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH

        print("横向间隙尺寸\n间隙1:%.2f\n间隙2:%.2f\n间隙3:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET, left_box_gap_width + param.LEFT_CENTER_OFFSET, center_box_gap_width), file=file)
        print("横向货物尺寸\n货物1:%.2f" % (left_box_width), file=file)
        print("库位中右侧位置的货物不存在，其他位置存在货物，库位右侧处于安全距离。", file=file)
        close_file_description(file)
        return
    
    if flag5 and flag4 and not flag3 and not flag2 and flag1:
        left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
        right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
        left_box_pixel_width = points5[1][0] - points5[0][0]
        right_box_pixel_width = points1[1][0] - points1[0][0]
        left_pillar_box_gap_pixel_width = points5[0][0] - left_pillar_right_x
        right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]
        left_box_gap_pixel_width = robust_points_gap_location(points4, img4)
        

        pillar_pixel_width = (left_pillar_pixel_width + right_pillar_pixel_width) / 2
        left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_box_gap_width = left_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        

        print("横向间隙尺寸\n间隙1:%.2f\n间隙2:%.2f\n间隙5:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET, left_box_gap_width + param.LEFT_CENTER_OFFSET, right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
        print("横向货物尺寸\n货物1:%.2f\n货物4:%.2f" % (left_box_width, right_box_width), file=file)
        print("库位中中间靠右侧位置的货物不存在，其他位置存在货物，库位中间处于安全距离。", file=file)
        close_file_description(file)
        return
    
    if flag5 and not flag4 and not flag3 and flag2 and flag1:
        left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
        right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
        left_box_pixel_width = points5[1][0] - points5[0][0]
        right_box_pixel_width = points1[1][0] - points1[0][0]
        left_pillar_box_gap_pixel_width = points5[0][0] - left_pillar_right_x
        right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]
        right_box_gap_pixel_width = robust_points_gap_location(points2, img2)
        

        pillar_pixel_width = (left_pillar_pixel_width + right_pillar_pixel_width) / 2
        left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_box_gap_width = right_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        

        print("横向间隙尺寸\n间隙1:%.2f\n间隙4:%.2f\n间隙5:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET, right_box_gap_width + param.RIGHT_CENTER_OFFSET, right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
        print("横向货物尺寸\n货物1:%.2f\n货物4:%.2f" % (left_box_width, right_box_width), file=file)
        print("库位中中间靠左侧位置的货物不存在，其他位置存在货物，库位中间处于安全距离。", file=file)
        close_file_description(file)
        return
    
    if not flag5 and not flag4 and flag3 and flag2 and flag1:
        right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
        right_box_pixel_width = points1[1][0] - points1[0][0]
        right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]
        right_box_gap_pixel_width = robust_points_gap_location(points2, img2)
        center_box_gap_pixel_width = robust_points_gap_location(points3, img3)

        pillar_pixel_width = right_pillar_pixel_width
        right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        right_box_gap_width = right_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
        center_box_gap_width = center_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH

        print("横向间隙尺寸\n间隙3:%.2f\n间隙4:%.2f\n间隙5:%.2f" % (center_box_gap_width, right_box_gap_width + param.RIGHT_CENTER_OFFSET, right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
        print("横向货物尺寸\n货物5:%.2f" % (right_box_width), file=file)
        print("库位中左侧位置的货物不存在，其他位置存在货物，库位左侧处于安全距离。", file=file)
        close_file_description(file)
        return


    left_pillar_pixel_width = left_pillar_right_x - left_pillar_left_x
    right_pillar_pixel_width = right_pillar_right_x - right_pillar_left_x
    left_box_pixel_width = points5[1][0] - points5[0][0]
    right_box_pixel_width = points1[1][0] - points1[0][0]
    left_pillar_box_gap_pixel_width = points5[0][0] - left_pillar_right_x
    right_pillar_box_gap_pixel_width = right_pillar_left_x - points1[1][0]

    left_box_gap_pixel_width = robust_points_gap_location(points4, img4)
    center_box_gap_pixel_width = robust_points_gap_location(points3, img3)
    right_box_gap_pixel_width = robust_points_gap_location(points2, img2)

    pillar_pixel_width = (left_pillar_pixel_width + right_pillar_pixel_width) / 2
    left_box_width = left_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    right_box_width = right_box_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    left_pillar_box_gap_width = left_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    right_pillar_box_gap_width = right_pillar_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    left_box_gap_width = left_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    center_box_gap_width = center_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH
    right_box_gap_width = right_box_gap_pixel_width / pillar_pixel_width * param.PILLAR_WIDTH

    print("横向间隙尺寸\n间隙1:%.2f\n间隙2:%.2f\n间隙3:%.2f\n间隙4:%.2f\n间隙5:%.2f" % (left_pillar_box_gap_width + param.LEFT_OFFSET, left_box_gap_width + param.LEFT_CENTER_OFFSET,\
                                                            center_box_gap_width, right_box_gap_width + param.RIGHT_CENTER_OFFSET, right_pillar_box_gap_width + param.RIGHT_OFFSET), file=file)
    print("横向货物尺寸\n货物1:%.2f\n货物4:%.2f" % (left_box_width, right_box_width), file=file)
    close_file_description(file)

    return


'''
    对一个库位的序列化图片进行测量
'''
def Serial_Images_Measurement(image_dir, save_dir, kuwei_type):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file = get_file_description(save_dir, "measurement.txt")

    if kuwei_type == param.KUWEI_TYPE_3:
        measure_kuwei_type_3(image_dir, save_dir, file)
    elif kuwei_type == param.KUWEI_TYPE_2:
        measure_kuwei_type_2(image_dir, save_dir, file)
    elif kuwei_type == param.KUWEI_TYPE_4:
        measure_kuwei_type_4(image_dir, save_dir, file)
    
    return


'''
    批量完成图片的序列化测量
'''
def batch_serial_measurement(data_src_dir, data_dst_dir):
    print("[WORK FLOW] Starting measuring horizontal size.")
    dirs = os.listdir(data_src_dir)
    for dir in dirs:
        if dir.endswith('.txt'):
            continue
        print("[INFO] Measuring " + dir + " horizontal size......")
        kuwei_type = param.KUWEI_TYPE_3
        try:
            Serial_Images_Measurement(os.path.join(data_src_dir, dir), os.path.join(data_dst_dir, dir), kuwei_type)
        except Exception as e:
            f = get_file_description(data_dst_dir, 'fail_log.txt')
            f.write(dir + " horizontal measurement fail! Please check this kuwei.")
            print("Exception info: " + repr(e), file=f)
            close_file_description(f)

    print("[INFO] Measurement task complete!")
    print("[WORK FLOW] Measuring horizontal size complete.")


'''
    批量完成图片的序列化测量(protocol)
'''
def batch_serial_measurement_protocol(data_src_dir, data_dst_dir):
    print("[WORK FLOW] Starting measuring horizontal size.")
    dirs = os.listdir(data_src_dir)
    for dir in dirs:
        if dir.endswith('.txt'):
            continue
        print("[INFO] Measuring " + dir + " horizontal size......")
        kuwei_type = str(dir.split('_')[-1])
        try:
            Serial_Images_Measurement(os.path.join(data_src_dir, dir), os.path.join(data_dst_dir, dir), kuwei_type)
        except Exception as e:
            f = get_file_description(data_dst_dir, 'fail_log.txt')
            f.write(dir + " horizontal measurement fail! Please check this kuwei.")
            print("Exception info: " + repr(e), file=f)
            close_file_description(f)

    print("[INFO] Measurement task complete!")
    print("[WORK FLOW] Measuring horizontal size complete.")


if __name__ == '__main__':
    # Serial_Images_Measurement(image_dir, save_dir)
    batch_serial_measurement(data_src_dir, data_dst_dir)
    # batch_serial_measurement_protocol(data_src_dir, data_dst_dir)