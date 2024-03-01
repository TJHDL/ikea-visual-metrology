# -*- coding: utf-8 -*-
# @Time    : 2022/7/29 16:42
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : config.py

import argparse

INPUT_IMAGE_SIZE = 512
# 0: confidence, 1: point_shape, 2: offset_x, 3: offset_y, 4: cos(direction),
# 5: sin(direction)
NUM_FEATURE_MAP_CHANNEL = 6
# image_size / 2^5 = 512 / 32 = 16
FEATURE_MAP_SIZE = 16
# Threshold used to filter marking points too close to image boundary
BOUNDARY_THRESH = 0.005 #0.005

# Thresholds to determine whether an detected point match ground truth.
SQUARED_DISTANCE_THRESH = 0.000277778  # 10 pixel in 600*600 image
DIRECTION_ANGLE_THRESH = 0.5235987755982988  # 30 degree in rad

VSLOT_MIN_DIST = 0.044771278151623496
VSLOT_MAX_DIST = 0.1099427457599304
HSLOT_MIN_DIST = 0.15057789144568634
HSLOT_MAX_DIST = 0.44449496544202816

BOX_MIN_DIST = 30000
BOX_MAX_DIST = 62500
BOX_MIN_X_DIST = 80
BOX_MAX_X_DIST = 256
BOX_MAX_VERTICAL_DIST = 80
# FEATURE_MAP_SIZE limit the size of block, each block's side if 1/16 of image
ADJOINT_RECT_SIDE_RATIO = 0.0625

SHORT_SEPARATOR_LENGTH = 0.199519231
LONG_SEPARATOR_LENGTH = 0.46875

# angle_prediction_error = 0.1384059287593468 collected from evaluate.py
BRIDGE_ANGLE_DIFF = 0.09757113548987695 + 0.1384059287593468
SEPARATOR_ANGLE_DIFF = 0.284967562063968 + 0.1384059287593468

SLOT_SUPPRESSION_DOT_PRODUCT_THRESH = 0.6   #0.8

# precision = 0.995585, recall = 0.995805
CONFID_THRESH_FOR_POINT = 0.11676871

def add_common_arguments(parser):
    """Add common arguments for training and inference."""
    parser.add_argument('--detector_weights',
                        help="The weights of pretrained detector.")
    parser.add_argument('--depth_factor', type=int, default=32,
                        help="Depth factor.")
    parser.add_argument('--disable_cuda', action='store_true',
                        help="Disable CUDA.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="Select which gpu to use.")

def get_parser_for_inference():
    """Return argument parser for inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['image', 'video', 'dir'],
                        help="Inference image or video or directory.")
    parser.add_argument('--video',
                        help="Video path if you choose to inference video.")
    parser.add_argument('--inference_slot', action='store_true',
                        help="Perform slot inference.")
    parser.add_argument('--thresh', type=float, default=0.5,
                        help="Detection threshold.")
    parser.add_argument('--save', action='store_true',
                        help="Save detection result to file.")
    add_common_arguments(parser)
    return parser