import numpy as np
import cv2
import torch
import config
from model import DirectionalPointDetector
from torchvision.transforms import ToTensor
from data import get_predicted_points, calc_point_squre_dist

BoxMPR_detector_weights = r'D:\ProjectCodes\VisionMeasurement\GapHeightMeasurement\checkpoints\dp_detector_59_dark.pth'   #开灯:r'checkpoints\dp_detector_799_v100.pth' 关灯:r'checkpoints\dp_detector_59_dark.pth'

BOXMPR_DEVICE = None
BOXMPR_MODEL = None

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
    pred_points = detect_marking_points(detector, image, 0.05, device)  #0.05
    return get_cornerPoints(image, pred_points)


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
        print("[INFO] 预加载BoxMPR模型")
        BOXMPR_DEVICE, BOXMPR_MODEL = get_device_and_BoxMPR_model()

    return detect_image(BOXMPR_MODEL, BOXMPR_DEVICE, image)


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

            # Step 7: bottom points filtration
            if point_i[1] >= 550 or point_j[1] >= 550:
                continue

            point_pairs.append((i, j))

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
            return True
    return False