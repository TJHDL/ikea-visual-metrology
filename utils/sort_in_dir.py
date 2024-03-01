# -*- coding: utf-8 -*-
# @Time    : 2023/9/2 10:45
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : sort_in_dir.py
import os
import cv2

file_dir = r'F:\ProjectImagesDataset\IKEA\20230729\horizontal_experiment\src_10\kuwei6'
dst_dir = r'F:\ProjectImagesDataset\IKEA\20230729\horizontal_experiment\src_10_reverse\kuwei6'

def generate_number(file_dir):
    for idx, file in enumerate(os.listdir(file_dir)):
        src = os.path.join(file_dir, file)
        number = "{:04d}".format(idx)
        dst = os.path.join(file_dir, str(number) + "." + file.split(".")[1])
        print("Saving to: " + dst)
        os.rename(src, dst, src_dir_fd=None, dst_dir_fd=None)

def reverse_number(file_dir, dst_dir):
    total_cnt = len(os.listdir(file_dir))
    for idx, file in enumerate(os.listdir(file_dir)):
        src = os.path.join(file_dir, file)
        img = cv2.imread(src)
        name = total_cnt - int(file.split(".")[0]) - 1
        number = "{:04d}".format(name)
        dst = os.path.join(dst_dir, str(number) + "." + file.split(".")[1])
        print("Saving to: " + dst)
        cv2.imwrite(dst, img)

if __name__ == '__main__':
    # generate_number(file_dir)
    reverse_number(file_dir, dst_dir)