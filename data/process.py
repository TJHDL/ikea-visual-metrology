"""Defines related function to process defined data structure."""
import math
import numpy as np
import torch
import config
import cv2
from .struct import MarkingPoint, detemine_point_shape


def non_maximum_suppression(pred_points):
    """Perform non-maxmum suppression on marking points."""
    suppressed = [False] * len(pred_points)
    for i in range(len(pred_points) - 1):
        for j in range(i + 1, len(pred_points)):
            i_x = pred_points[i][1].x
            i_y = pred_points[i][1].y
            j_x = pred_points[j][1].x
            j_y = pred_points[j][1].y
            # 0.0625 = 1 / 16
            # if abs(j_x - i_x) < 0.0625 and abs(j_y - i_y) < 0.0625:
            if (abs(j_x - i_x) < 0.015 and abs(j_y - i_y) < 0.015) or (abs(j_x - i_x) < 0.01 and abs(j_y - i_y) < 0.08):
                idx = i if pred_points[i][0] < pred_points[j][0] else j
                suppressed[idx] = True
    if any(suppressed):
        unsupres_pred_points = []
        for i, supres in enumerate(suppressed):
            if not supres:
                unsupres_pred_points.append(pred_points[i])
        return unsupres_pred_points
    return pred_points


def get_predicted_points(prediction, thresh):
    """Get marking points from one predicted feature map."""
    assert isinstance(prediction, torch.Tensor)
    predicted_points = []
    prediction = prediction.detach().cpu().numpy()
    for i in range(prediction.shape[1]):
        for j in range(prediction.shape[2]):
            if prediction[0, i, j] >= thresh:
                xval = (j + prediction[2, i, j]) / prediction.shape[2]
                yval = (i + prediction[3, i, j]) / prediction.shape[1]
                if not (config.BOUNDARY_THRESH <= xval <= 1-config.BOUNDARY_THRESH
                        and config.BOUNDARY_THRESH <= yval <= 1-config.BOUNDARY_THRESH):
                    continue
                cos_value = prediction[4, i, j]
                sin_value = prediction[5, i, j]
                direction = math.atan2(sin_value, cos_value)
                marking_point = MarkingPoint(
                    xval, yval, direction, prediction[1, i, j])
                predicted_points.append((prediction[0, i, j], marking_point))
    return non_maximum_suppression(predicted_points)


def pass_through_third_point(marking_points, i, j):
    """See whether the line between two points pass through a third point."""
    x_1 = marking_points[i][0]
    y_1 = marking_points[i][1]
    x_2 = marking_points[j][0]
    y_2 = marking_points[j][1]
    for point_idx in range(len(marking_points)):
        if point_idx == i or point_idx == j:
            continue
        point = marking_points[point_idx]
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
    判别局部图像是否包含角点
'''
def detect_area_harris_corner(point_a, point_b, image):
    """See whether the adjoint area of detected points overlaps the corner of the box."""
    img_size = image.shape[0]
    deviation = math.floor(img_size * config.ADJOINT_RECT_SIDE_RATIO / 2)
    area_a = image[max(point_a[1]-deviation, 0):min(point_a[1]+deviation, img_size), \
        max(point_a[0]-deviation, 0):min(point_a[0]+deviation, img_size)]
    area_b = image[max(point_b[1] - deviation, 0):min(point_b[1] + deviation, img_size), \
        max(point_b[0] - deviation, 0):min(point_b[0] + deviation, img_size)]

    gray_a = cv2.cvtColor(area_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(area_b, cv2.COLOR_BGR2GRAY)
    blur_a = cv2.GaussianBlur(gray_a, (5, 5), 0)
    blur_b = cv2.GaussianBlur(gray_b, (5, 5), 0)
    retval_a, dst_a = cv2.threshold(blur_a, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    retval_b, dst_b = cv2.threshold(blur_b, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(dst_a, np.flipud(dst_b))

    height = mask.shape[0]
    width = mask.shape[1]
    if (mask[height-1, 0] == 255 and mask[0, width-1] ==255) or (mask[0, 0] == 255 and mask[height-1, width-1]):
        return False

    return True

def pair_marking_points(point_a, point_b):
    """See whether two marking points form a slot."""
    vector_ab = np.array([point_b.x - point_a.x, point_b.y - point_a.y])
    vector_ab = vector_ab / np.linalg.norm(vector_ab)
    point_shape_a = detemine_point_shape(point_a, vector_ab)
    point_shape_b = detemine_point_shape(point_b, -vector_ab)
    if point_shape_a.value == 0 or point_shape_b.value == 0:
        return 0
    if point_shape_a.value == 3 and point_shape_b.value == 3:
        return 0
    if point_shape_a.value > 3 and point_shape_b.value > 3:
        return 0
    if point_shape_a.value < 3 and point_shape_b.value < 3:
        return 0
    if point_shape_a.value != 3:
        if point_shape_a.value > 3:
            return 1
        if point_shape_a.value < 3:
            return -1
    if point_shape_a.value == 3:
        if point_shape_b.value < 3:
            return 1
        if point_shape_b.value > 3:
            return -1
