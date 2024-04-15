import argparse

# horizontal parameters
LEFT_OFFSET = 1.6   #cm 1.6
RIGHT_OFFSET = -1.4 #cm -1.4
LEFT_CENTER_OFFSET = 0.1    #cm 0.1
RIGHT_CENTER_OFFSET = 0.9   #cm 0.9
PILLAR_WIDTH = 11.8 #cm

'''
    | 库位宽度对照表 |
    -----------------
    | 库位类型 | 宽度 |
    |4货物库位 | 428cm |
    |3货物库位 | 325cm |
    |2货物库位 | 222cm |
'''
KUWEI_WIDTH = {4:428, 3:325, 2:222}   #cm

# vertical parameters
RED_WIDTH = 10 #cm
TIEPIAN_WIDTH = 3 #cm
FLOOR_NUM = 3   #3
FLOOR_HEIGHT = 140  #cm
CAR_HEIGHT = 87 #cm

# 特殊高度货架号
SPECIAL_HUOJIA = [106, 107, 108, 109, 601, 602, 603, 101, 110, 111, 114, 117]
SPECIAL_HUOJIA_DELTA_HEIGHT = 20    #cm

'''
    |   无人机高度对照表   |
    -----------------------
    |层号|真实高度|参数高度|
    |2层 | 100cm | 125cm  |
    |3层 | 260cm | 260cm  |
    |4层 | 400cm | 400cm  |
    |5层 | 540cm | 540cm  |
    |6层 | 680cm | 680cm  |
'''
UAV_HEIGHT = { 2 : 125,
               3 : 260,
               4 : 400,
               5 : 540,
               6 : 680
               }
'''
    将上方横梁的上边沿当作地面，参照论文把上下调转过来，此数值需要根据无人机的飞行高度、货架单层高度进行计算估计
'''
H_CAMERA = FLOOR_NUM * FLOOR_HEIGHT - (CAR_HEIGHT + UAV_HEIGHT[FLOOR_NUM]) - TIEPIAN_WIDTH #cm

KUWEI_TYPE_2 = 2
KUWEI_TYPE_3 = 3
KUWEI_TYPE_4 = 4
KUWEI_TYPE_THRESHOLD_CHOICES = {2:25, 3:40, 4:55}
MAX_KUWEI_IMAGES_COUNT = 160

START_POS_RANGE_LEFT = 0.686    # 12 / 17
START_POS_RANGE_RIGHT = 0.785   # 13 / 17
END_POS_RANGE_LEFT = 0.215  # 4 / 17
END_POS_RANGE_RIGHT = 0.314  # 5 / 17

KUWEI_TYPE_3_FIG2_MAX_OFFSET = 16
KUWEI_TYPE_3_FIG2_MIN_OFFSET = 6
KUWEI_TYPE_3_FIG4_MAX_OFFSET = 18
KUWEI_TYPE_3_FIG4_MIN_OFFSET = 8

KUWEI_TYPE_2_FIG2_LR_OFFSET = 8

KUWEI_TYPE_4_FIG2_OFFSET = 13
KUWEI_TYPE_4_FIG6_OFFSET = 13
KUWEI_TYPE_4_FIG2_LR_OFFSET = 5
KUWEI_TYPE_4_FIG4_LR_OFFSET = 5
KUWEI_TYPE_4_FIG6_LR_OFFSET = 5

CENTER_POINT_LEFT_THRESHOLD = 0.35
CENTER_POINT_RIGHT_THRESHOLD = 0.65

KUWEI_TYPE_IMAGES_NUM_THRESHOLD = {'2-3':50, '3-4':95}

# 正无穷大
POSITIVE_INFINITY = float('inf')
# 负无穷大
NEGATIVE_INFINITY = float('-inf')

HORIZONTAL_SAFE_THRESHOLD = 2   #cm
VERTICAL_SAFE_THRESHOLD = 5     #cm

def get_parser_for_measurement():
    parser = argparse.ArgumentParser(description='Arguments for spliting kuwei and measurement')
    parser.add_argument('--img_dir', type=str, default=r'C:\Users\95725\Desktop\rtsp_picture_20240322\floor4',
                        help="Original rtsp video stream frames' directory.")
    parser.add_argument('--src_dir', type=str, default=r'C:\Users\95725\Desktop\src',
                        help="Splited kuweis storage directory.")
    parser.add_argument('--dst_dir', type=str, default=r'C:\Users\95725\Desktop\dst',
                        help="Measurement results storage directory")
    parser.add_argument('--floor', type=int,
                        help="Floor number of captured images.")
    parser.add_argument('--kuwei_type', type=int, choices=[3, 2], default=3,
                        help="Measured kuwei type.")
    parser.add_argument('--use_protocol', type=bool, default=False,
                        help="Whether use protocol to save images or not.")
    parser.add_argument('--xls_file', type=str, default=r'report\407-03-00-60_20231129.xls',
                        help="Measurement result report excel file path.")
    
    return parser