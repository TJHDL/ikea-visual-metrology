import argparse
import socket
import threading
import time
import math
import logging
import os
import datetime

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets_measurement import *
from utils.utils import *

import utils.protocol as protocol

LOG_PATH = r'/home/nvidia/YIJIA/yolov5test2/logs'
logger = None

# 定义一个全局变量，用于标识线程是否应该继续运行
RUNNING = True

# 定义一个函数，用于接收来自服务端的消息
def receive_messages(client_socket):
    global RUNNING, logger
    while RUNNING:
        try:
            # 接收消息
            message = client_socket.recv(1024).decode('utf-8')
            if message:
                print("Received: ", message)
                logger.info(message)
                infos = message.split('/')
                protocol.HUOJIA, protocol.KUWEI, protocol.FLOOR, protocol.SPLIT = infos[0], infos[1], infos[2], infos[3]
                protocol.SPLIT = int(protocol.SPLIT)
        except Exception as e:
            print("Error receiving message:", str(e))
            break

        # 控制接收消息的频率
        time.sleep(1 / 20)  # 20Hz


def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.FileHandler(log_file)
    handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def log_request(logger, timestamp, guangbanYaw, ledYaw, ratio):
    logger.info(f'时间戳: {timestamp}, 光斑偏航角估计: {guangbanYaw}, LED偏航角估计: {ledYaw}, 标签宽度/长度:{ratio}')


def lightDetect(img): #通过光斑在图像中的位置以及偏航角
    # Step1: 二值化处理确定光斑区域
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 187, 255, cv2.THRESH_BINARY)

    # Step2: 按比例缩小原图加速运算，提取白色像素坐标
    minimize_ratio = 1
    dst_mini = cv2.resize(dst, (int(dst.shape[1] / minimize_ratio), int(dst.shape[0] / minimize_ratio)))
    # iterator2: np.where计算迅速，1088x720耗时约0.006s
    data = np.where(dst_mini == 255)[1]
    
    #print("白色像素:" + str(len(data)))
    # Safe Module 防止视野中无光斑或过曝
    if len(data) <= 300 or len(data) >= 0.2 * dst_mini.shape[0] * dst_mini.shape[1]:
        print("Trouble: white pixels too less or too much! x_ratio=0, alpha=0.")
        return 0

    # Step3: 直接对横坐标求均值
    x_pos = np.mean(data)
    # cv2.circle(img, (int(x * minimize_ratio), int(dst.shape[0] / 2)), 100, (2, 30, 200), 10)

    # Step4: 求光斑中心横坐标相对于图片宽度一半的比例（此处定义的坐标系原点位置为[width/2, 0]）,该比例有正负
    center = int(dst_mini.shape[1] / 2)
    x_ratio = (x_pos - center) / center
    #print("光斑位于x轴方向的比例: " + str(x_ratio))

    # Step5: 根据关系 tanα = tan42° * x_ratio 计算偏航角alpha,单位°
    #alpha = math.atan(math.tan(math.radians(42)) * x_ratio)
    alpha = math.degrees(math.atan(math.tan(math.radians(42)) * x_ratio))
    # print("估计的偏航角alpha: " + str(alpha))

    return alpha

def led_detect_angles2(image): #灯条检测偏航角

    # 读取图像
    # bgr_img = cv2.imread(image)
    # bgr图像转化为灰度图像
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 灰度图像转化为二值图像
    th, binary = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
    # 轮廓函数返回值为两个，第一个是轮廓contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓

    # 拟合直线时所需的行列变量
    rows, cols = image.shape[:2]

    angle_delta = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 100: #对于1080P图像； 若对象为5200*3900照片，可取1000
            # print(cv2.contourArea(cnt))
            # 最小矩形拟合
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            is_shorter = math.sqrt((box[0, 0] - box[1, 0]) ** 2 + (box[0, 1] - box[1, 1]) ** 2)
            is_longer = math.sqrt((box[0, 0] - box[3, 0]) ** 2 + (box[0, 1] - box[3, 1]) ** 2)

            if is_longer >= is_shorter:
                ratio_bounding = is_longer / is_shorter
            else:
                ratio_bounding = is_shorter / is_longer



            if ratio_bounding < 45 and ratio_bounding > 20:  # 实际值35左右
                #角点拟合
                [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                # 计算斜率倾斜角
                angle = math.degrees(math.atan2((righty - lefty), (cols - 1 - 0)))
                if angle > 0 and angle < 90:
                    angle = 90 - angle
                if angle < 0 and angle > -90:
                    angle = -1.0 * (90 - angle)
                if angle < -90 and angle > -180:
                    angle = -1.0 * (180 + angle)

                if -30 <= angle <= 30:
                    angle_delta.append(angle)
                else:
                    print("angle > 30 ",angle)
                    return 0
            else:
                return 0

    print("--------------------size: " + str(len(angle_delta)))
    print("--------------------angle_delta: " + str(angle_delta))
    if len(angle_delta) == 1:
        angle_data_average = angle_delta[0]
    else:
        angle_data_average = 0

    if angle_data_average != 0:
        print("预测的偏航角: " + str(angle_data_average))

    return angle_data_average


## spot angle detect

fx = 686.1758
fy = 603.8131
spot_len = 0.62
kp = 1.2

def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def isLine(a, b, c):
    if b[0] - a[0] != 0 and c[0] - a[0] != 0:
        slope1 = (b[1] - a[1]) / (b[0] - a[0])
        slope2 = (c[1] - a[1]) / (c[0] - a[0])

        if abs(slope1 - slope2) < 0.05:
            return True
        else: 
            return False
    elif b[0] - a[0] == 0 and c[0] - a[0] == 0:
        return True
    else:
        return False

def isNotSamePoint(a, b, c):
    disab = distance(a, b)
    disac = distance(a, c)
    disbc = distance(b ,c)
    if disab > 10 and disac > 10 and disbc > 10:
        return True
    else:
        return False

def isTarget(a, b, c):
    disab = distance(a, b)
    disac = distance(a, c)
    disbc = distance(b, c)
    edges = [disab, disac, disbc]
    edges.sort()    
    # print("edges: ", edges[1] / edges[0])
    if edges[0] != 0 and edges[1] / edges[0] > 3.0 and edges[1] / edges[0] < 3.3:
        return True
    
    return False

def isSameSize(r1, r2, r3):
    if r1 > 5 and r2 > 5 and r3 > 5 and abs(r1 - r2) < 10 and abs(r2 - r3) < 10 and abs(r3 - r1) < 10:
        return True
    return False

def counterclockwise_sort_points(v_points):
    cnt = len(v_points)
    if cnt < 3:
        return
    
    center = np.mean(v_points, axis=0)
    
    def point_cmp(a, b, center):
        if abs(a[0] - b[0]) < 10:
            return a[1] > b[1]
        return a[0] > b[0]
    
    for i in range(cnt - 1):
        for j in range(cnt - i - 1):
            if point_cmp(v_points[j], v_points[j + 1], center):
                v_points[j], v_points[j + 1] = v_points[j + 1], v_points[j]

def spot_detect_angles(img):
    rows, cols = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    structureElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode_image = cv2.erode(thresh, structureElement)
    dilate_image = cv2.dilate(erode_image, structureElement, iterations=1)
    contours, _ = cv2.findContours(dilate_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    radius = []
    points_ = []
    for contour in contours:
        center, _radius = cv2.minEnclosingCircle(contour)
        center = (int(center[0]), int(center[1]))
        points.append(center)
        radius.append(_radius)

    if len(points) >= 3 and len(points) <= 12:
        for i in range(len(points) - 2):
            for j in range(i + 1, len(points) - 1):
                for k in range(j + 1, len(points)):
                        if isLine(points[i], points[j], points[k]) and isTarget(points[i], points[j], points[k]) and isSameSize(radius[i], radius[j], radius[k]) and isNotSamePoint(points[i], points[j], points[k]):
                            points_.extend([points[i], points[j], points[k]])
    else:
        print("size: " + str(len(points)))
        return 0, 0, 0, 0

    if len(points_) >= 3:
        points_2 = []
        points_2.extend([points_[0], points_[1], points_[2]])
        counterclockwise_sort_points(points_2)
            
        for i in range(len(points_2)):
            # print(points_2[i][0], points_2[i][1])
            cv2.line(img, points_2[i], points_2[(i + 1) % len(points_2)], (0, 255, 0), 1, cv2.LINE_AA)

        isLongPoint = points_2[0]
        isShortPoint = points_2[2]
        AB = distance(points_2[0], points_2[1])
        BC = distance(points_2[1], points_2[2])    
        if AB > BC:
            isLongPoint = points_2[0]
            isShortPoint = points_2[2]
        else:
            isLongPoint = points_2[2]
            isShortPoint = points_2[0]
        
        angle = math.atan2(isLongPoint[1] - isShortPoint[1], isLongPoint[0] - isShortPoint[0])
        # spot_y = spot_len * math.sin(abs(angle))
        height = (kp * fy) * spot_len / distance(isLongPoint, isShortPoint)     #使用z的时候慎重，需与雷达坐标对齐
        center_x = (isLongPoint[0] + isShortPoint[0]) / 2
        center_y = (isLongPoint[1] + isShortPoint[1]) / 2
        pos_x = (center_x - cols / 2) / (kp * fx) * height      #使用x的时候慎重，需与雷达坐标对齐
        pos_y = -(center_y - rows / 2) / (kp * fy) * height     #使用y的时候慎重，需与雷达坐标对齐
        pos_angle = angle * 180 / 3.1415926
            
        pos_angle = -pos_angle
        
        if abs(pos_angle) < 30:
            return pos_x, pos_y, height, pos_angle
        else:
            return 0, 0, 0, 0

    else:
        print("points_num:", len(points_))
        return 0, 0, 0, 0

## spot angle detect    

def led_detect_angles(image):  #通过灯带检测偏航角

    # 读取图像
    # bgr_img = cv2.imread(image)
    # bgr图像转化为灰度图像
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 灰度图像转化为二值图像
    th, binary = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
    # 轮廓函数返回值为两个，第一个是轮廓contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓

    # 拟合直线时所需的行列变量
    rows, cols = image.shape[:2]

    angle_delta = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            # 最小矩形拟合
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            is_shorter = math.sqrt((box[0, 0] - box[1, 0]) ** 2 + (box[0, 1] - box[1, 1]) ** 2)
            is_longer = math.sqrt((box[0, 0] - box[3, 0]) ** 2 + (box[0, 1] - box[3, 1]) ** 2)

            if is_longer >= is_shorter:
                ratio_bounding = is_longer / is_shorter
            else:
                ratio_bounding = is_shorter / is_longer


            if ratio_bounding < 10 and ratio_bounding > 3:
                #角点拟合
                [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                # 计算斜率倾斜角
                angle = math.degrees(math.atan2((righty - lefty), (cols - 1 - 0)))
                if angle > 0 and angle < 90:
                    angle = 90 - angle
                if angle < 0 and angle > -90:
                    angle = -1.0 * (90 - angle)
                if angle < -90 and angle > -180:
                    angle = -1.0 * (180 + angle)

                if -30 <= angle <= 30:
                    angle_delta.append(angle)
                else:
                    # print("angle > 30 ",angle)
                    return 0
            else:
                # print("ratio bounding error")
                return 0

    if len(angle_delta) == 2:
        angle_data_average = (angle_delta[0] + angle_delta[1]) / 2
    else:
        angle_data_average = 0
    
    if angle_data_average != 0:
        print("预测的偏航角: " + str(angle_data_average))

    return angle_data_average



def detect(save_img=False):  #检测标签
    global conn, center_x_center, det_number_det, center_y_center, logger
    frequency, label_cnt, yaw_list = 10, 0, []

    log_file = os.path.join(LOG_PATH, "log_" + str(datetime.datetime.now().date()) + ".txt")
    logger = setup_logger("yaw_estimation", log_file)

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize 初始化
    device = torch_utils.select_device(opt.device)  #Using CUDA device0 _CudaDeviceProperties(name='Orin', total_memory=30622MB)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model,Fusing layers... Model Summary: 284 layers, 8.83906e+07 parameters, 8.45317e+07 gradients
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader 加载图像数据
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)   #1/1: rtsp://yijia:123@192.168.3.154:8554/streaming/live/1...  success (1088x720 at 31.00 FPS).
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    n = 0
    while True:
        # if dataset.imgs is not None:
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                # alpha_led = led_detect_angles2(im0)
                x_led, y_led, z_led, alpha_led = spot_detect_angles(im0)
                #print('im01:' + str(im0.shape[0]) + "," + str(im0.shape[1]))
                if det is None:
                    det_number_det = 0

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    det_number_det = n

                    # Write results
                    save_txt = True
                    view_img = True
                    center_x_list=[] #标签中心点在图片的横坐标信息
                    center_y_list=[] #标签中心点在图片的纵坐标信息
                    width2=[]
                    height2=[]
                    count_x=0
                    state=0

                    for *xyxy, conf, cls in det:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        # 滤除横梁上的标签干扰
                        box_width = xywh[2] * im0.shape[1]
                        box_height = xywh[3] * im0.shape[0]
                        width2.append(box_width)
                        height2.append(box_height)
                        # print('box_width / box_height :',box_width / box_height )
                        #if box_width / box_height >= 2.5:
                            #print("Detect label in crossbeam, filter it!")
                            #continue
                        #print("宽度:长度 = " + str(box_width/box_height))
                        if box_width / box_height <1.5:
                            if save_img or view_img:  # Add bbox to image
                                label = '%s %.2f' % (names[int(cls)], conf)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                center_x_center = xywh[0] + 0.5 * xywh[2]
                                center_y_center = xywh[1] + 0.5 * xywh[3]

                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format


                                #print('center_x:', center_x_center)
                                #print('center_y:', center_y_center)

                                center_x_list.append(center_x_center)
                                center_y_list.append(center_y_center)

                    # client
                    for count_x in range(len(center_x_list)):
                        # continue
                        if 0.48<center_x_list[count_x]<0.78: #标签中心点在图片的横坐标在0.48-0.78范围内时
                            # print('-------------------------------------aaaaa-------------------------------------')
                            alpha_guangban = lightDetect(im0)
                            ratio2 = width2[count_x] / height2[count_x]
                            log_request(logger, time.time(), alpha_guangban, alpha_led, ratio2)
                            # sk.send(bytes('1/' + str(center_x_list[count_x]) + '/' + str(center_y_list[count_x]) + '/0/0\n', encoding='utf8')) #发送标签位置和偏航角给AGV
                            sk.send(bytes('1/' + str(center_x_list[count_x]) + '/' + str(center_y_list[count_x]) + '/' + '0' + '/' + str(alpha_led) + '\n', encoding='utf8'))  # 发送标签位置和偏航角给AGV
                            state=2
                    if state == 0:
                        # print('-----------------bbbbb-------------------')
                        log_request(logger, time.time(), 0, alpha_led, 0)
                        # sk.send(bytes('0/0/0/0/0\n', encoding='utf8')) #发送数据给AGV
                        sk.send(bytes('0/0/0/0/' + str(alpha_led) + '\n', encoding='utf8'))  # 发送数据给AGV
                    time.sleep(0.05)

                else:
                    log_request(logger, time.time(), 0, alpha_led, 0)
                    # sk.send(bytes('0/0/0/0/0\n', encoding='utf8'))
                    sk.send(bytes('0/0/0/0/' + str(alpha_led) + '\n', encoding='utf8')) #发送数据给AGV
                    # print('-----------------ccccc-----------------')

                # Print time (inference + NMS)
                # print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    # print('开始取图')
                    cv2.imshow(p, im0) #显示检测后的视频流数据
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)


if __name__ == '__main__':
    # global conn, center_x
    #参数设置，--weights是模型参数， --source是传入图像/图像位置/rtsp视频流等， --output是检测之后图片以及标签位置信息的txt保存位置
    #--img-size指的是模型在检测图片前会把图片resize成固定尺寸，再放进网络，并不是指最终结果热size成固定大小
    #--conf-thres置信度的阈值，--iou-thres是调节iou阈值
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--weights', nargs='+', type=str,
            
                        default='/home/nvidia/YIJIA/yolov5test2/runs/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='rtsp://yijia:123@192.168.3.233:8554/streaming/live/1',help='source')
    parser.add_argument('--output', type=str, default='inference/output_video', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    # client客户端 与agv建立连接
    try:
        client_socket = socket.socket()
        client_socket.connect(('192.168.3.217', 1234))
        print('连接成功！')

        # 创建一个线程来接收消息
        receive_thread = threading.Thread(target=receive_messages, args=(client_socket,))
        receive_thread.start()

        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                    print('进入detcet')
                    detect()
                    create_pretrained(opt.weights, opt.weights)
            else:
                detect()
    except Exception as e:
        print('Error: ', str(e))
    finally:
        # client
        client_socket.close()  # 关闭资源
        RUNNING = False

    # sever
    # conn.close()  # 关闭资源
    # sk.close()  # 关闭资源
