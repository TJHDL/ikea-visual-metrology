# -*- coding: utf-8 -*-
# @Time    : 2023/11/1 17:03
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : file_util.py
import os
import cv2

'''
    将获取的关键帧保存至目标位置
'''
def save_key_frames(key_frames, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for idx, image in enumerate(key_frames):
        cv2.imwrite(os.path.join(save_dir, str(idx + 1) + '.jpg'), image)


'''
    获取有序命名的文件夹内序号最大的文件的序号值
'''
def get_head_tail_sorted_number(image_dir):
    files = os.listdir(image_dir)
    max_number = 0
    min_number = len(files)
    for file in files:
        number = int(file.split('.')[0])    # without timestamp
        # number = int(file.split('.')[0].split('_')[1])  # with timestamp
        max_number = max(max_number, number)
        min_number = min(min_number, number)

    return min_number, max_number

'''
    创建一个用于写入测量结果的txt文件
'''
def get_file_description(save_dir, txt_name):
    if not os.path.exists(os.path.join(save_dir, txt_name)):
        with open(os.path.join(save_dir, txt_name), "w") as file:
            file.write("Measurement Result\n")
    file = open(os.path.join(save_dir, txt_name), "a", encoding="utf-8")

    return file

'''
    关闭文件描述符
'''
def close_file_description(file):
    file.flush()
    file.close()

def rename_all_files(image_dir):
    for filename in os.listdir(image_dir):
        print("file name: ", filename)
        # if not os.path.isfile(filename):  # 判断是否为文件而非文件夹
        #     continue

        old_path = os.path.join(image_dir, filename)  # 构建原始文件的完整路径
        prefix, suffix = filename.split('.')[0], filename.split('.')[1]
        new_name = [prefix.split('_')[1], suffix]
        new_name = '.'.join(new_name)
        new_path = os.path.join(image_dir, new_name)  # 构建新文件的完整路径

        try:
            os.rename(old_path, new_path)  # 调用系统函数进行重命名操作
            print("已经将文件 {} 重命名为 {}".format(filename, new_name))
        except Exception as e:
            print("无法重命名文件 {}: {}".format(filename, str(e)))

if __name__ == '__main__':
    image_dir = r'C:\Users\95725\Desktop\rtsp_picture_20240119_407'
    rename_all_files(image_dir)