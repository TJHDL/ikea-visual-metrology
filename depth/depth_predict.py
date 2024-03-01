# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 19:59
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : predict.py

import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from depth.utils.utils import *
from depth.models.net import SARPN_APP
from depth.dataloader import nyudv2_dataloader
import matplotlib.pyplot as plt

BATCH_SIZE = 1
IMAGE_NAME = '5.jpg'
ROOT_PATH = r'D:\ProjectCodes\VisionMeasurement\0926test\src\kuwei6'
PRETRAINED_DIR = r'D:\ProjectCodes\VisionMeasurement\SARPNDepthEstimation\pretrained_dir'
LOADCKPT = r'D:\ProjectCodes\VisionMeasurement\SARPNDepthEstimation\my_checkpoint\SARPN_checkpoints_240.pth.tar'
INTRINSIC = [[2421.3,   11.6217,  2655.2],
            [0,   2442.4, 2110.6],
            [0,    0, 1]]

def load_model(batch_size, image_name, root_path):
    # img = cv2.imread(os.path.join(ROOT_PATH, IMAGE_NAME))
    TestImgLoader = nyudv2_dataloader.getSingleData_IKEA(batch_size, image_name, root_path)
    model = SARPN_APP(PRETRAINED_DIR)
    model = nn.DataParallel(model)
    # model.cuda()

    # load test model
    if LOADCKPT is not None:
        # print("loading model {}".format(LOADCKPT))
        state_dict = torch.load(LOADCKPT, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    else:
        print("You have not loaded any models.")

    return TestImgLoader, model

def depth_estimation(batch_size, image_name, root_path):
    TestImgLoader, model = load_model(batch_size, image_name, root_path)

    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            image = sample['image']
            # image = image.cuda()

            image = torch.autograd.Variable(image)

            start = time.time()
            pred_depth = model(image)
            end = time.time()
            running_time = end - start
            output = torch.nn.functional.interpolate(pred_depth[-1], size=[480, 640], mode='bilinear',
                                                     align_corners=True) / 10
            output = np.squeeze(output.cpu().detach().numpy())
            #   可视化深度信息
            # plt.imshow(output)
            # plt.savefig('prediction.jpg')
            output = output * 1000  # 单位: cm
            output[np.isnan(output)] = output.mean()
            output[np.isinf(output)] = output.mean()
            # print(output.mean())
            # print(output.max())
            # print(output.min())

    return output

def plane_to_pointcloud(point, depth):
    fx, fy = INTRINSIC[0][0], INTRINSIC[1][1]
    cx, cy = INTRINSIC[0][2], INTRINSIC[1][2]
    x = int((point[0] - cx) / fx * depth)
    y = int((point[1] - cy) / fy * depth)
    z = int(depth)
    return x, y, z


if __name__ == '__main__':
    output = depth_estimation(BATCH_SIZE, IMAGE_NAME, ROOT_PATH)