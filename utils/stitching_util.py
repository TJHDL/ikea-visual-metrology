# -*- coding: utf-8 -*-
# @Time    : 2023/8/13 22:42
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : stitching_util.py

from utils.panorama import Panaroma
import imutils
import cv2
import os
import numpy as np
from PIL import Image

image_dir = r'D:\ProjectCodes\VisionMeasurement\stiching_test\img_dir\kuwei6'
save_dir = r'D:\ProjectCodes\VisionMeasurement\stiching_test\result_dir\kuwei6'

def crop_image(img):
    # 全景图轮廓提取
    stitched = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 轮廓最小正矩形
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(cnts[0])  # 取出list中的轮廓二值图，类型为numpy.ndarray
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # 腐蚀处理，直到minRect的像素值都为0
    minRect = mask.copy()
    sub = mask.copy()
    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)
    stitched = stitched[y:y + h, x:x + w]
    # cv2.imshow("去黑边结果", stitched)

    imageRGB = cv2.cvtColor(stitched.astype('uint8'), cv2.COLOR_BGR2RGB)
    stitched = Image.fromarray(imageRGB)
    # stitched.save(os.path.join(save_dir, 'result.jpg'))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    stitched_img = cv2.cvtColor(np.asarray(stitched), cv2.COLOR_RGB2BGR)
    # stitched_img = cv2.resize(stitched_img, (512 * 3, 512))

    # cv2.namedWindow("stitched_img", 0)
    # cv2.imshow("stitched_img", stitched_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return stitched_img

def stitch_image(image_dir):
    print("The number of images you want to concantenate:")
    no_of_images = int(len(os.listdir(image_dir)))
    print(no_of_images)
    print("Feed the image name in order of left to right in way of concantenation:")
    images = []

    for i in range(no_of_images, 0, -1):
        images.append(cv2.resize(cv2.imread(os.path.join(image_dir, str(i) + '.jpg')), (512, 512)))

    panaroma = Panaroma()
    H_list = []
    if no_of_images == 2:
        (result, matched_points, H) = panaroma.image_stitch([images[0], images[1]], match_status=True)
        H_list.append(H)
    else:
        (result, matched_points, H) = panaroma.image_stitch([images[no_of_images - 2], images[no_of_images - 1]],
                                                         match_status=True)
        H_list.append(H)
        for i in range(no_of_images - 2):
            (result, matched_points, H) = panaroma.image_stitch([images[no_of_images - i - 3], result], match_status=True)
            H_list.append(H)

    # to write the images
    # cv2.imwrite(os.path.join(save_dir, "Matched_points.jpg"), matched_points)
    # cv2.imwrite(os.path.join(save_dir, "Panorama_image.jpg"), result)

    # cv2.imshow("Keypoint Matches", matched_points)
    # cv2.namedWindow("Panorama", 0)
    # cv2.imshow("Panorama", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    stitched_img = crop_image(result)
    return stitched_img, H_list

if __name__ == '__main__':
    stitch_image(image_dir)