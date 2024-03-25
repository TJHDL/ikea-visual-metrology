import numpy as np
import os
import cv2
from depth.depth_predict import depth_estimation

STANDARD_DEPTH = 88.67

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