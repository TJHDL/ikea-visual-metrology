

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
    |3货物库位 | 325cm |
    |2货物库位 | 222cm |
'''
KUWEI_WIDTH = 325   #cm

# vertical parameters
RED_WIDTH = 10 #cm
TIEPIAN_WIDTH = 3 #cm
FLOOR_NUM = 3   #3
FLOOR_HEIGHT = 140  #cm
CAR_HEIGHT = 87 #cm

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
UAV_HEIGHT = 260    #cm 2层若以100高度飞应取125计算 3层对应260 4层对应400
H_CAMERA = FLOOR_NUM * FLOOR_HEIGHT - (CAR_HEIGHT + UAV_HEIGHT) - TIEPIAN_WIDTH #cm   将上方横梁的上边沿当作地面，参照论文把上下调转过来，此数值需要根据无人机的飞行高度、货架单层高度进行计算估计