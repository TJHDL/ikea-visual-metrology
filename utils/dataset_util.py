# -*- coding: utf-8 -*-
# @Time    : 2023/11/26 15:45
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : dataset_util.py
import cv2
import os

src_dir = r'C:\Users\95725\Desktop\rtsp_picture_1009_407_floor3'
dst_dir = r'C:\Users\95725\Desktop\semantic_dataset_rtsp_1009\img_data'

def resize_and_rename(src_dir, dst_dir):
    image_list = os.listdir(src_dir)

    number = 1
    for image_name in image_list:
        if number % 5 == 0:
            image = cv2.imread(os.path.join(src_dir, image_name))
            # image = cv2.resize(image, (3000, 3000))
            save_name = "rtsp_1009_floor3_" + image_name
            save_path = os.path.join(dst_dir, save_name)
            cv2.imwrite(save_path, image)
            print("Saving " + save_path)
            number = 0
        number += 1

if __name__ == '__main__':
    resize_and_rename(src_dir, dst_dir)