# -*- coding: utf-8 -*-
# @Time    : 2023/5/30 22:23
# @Author  : Dingliang Huang
# @Affi    : Tongji University BISL Group
# @File    : crop_image.py
import os
import cv2

image_dir = r'F:\ProjectImagesDataset\IKEA\Measurement\test_raw\image2'
image_name = r'IMG_0011.jpg'
save_dir = r'F:\ProjectImagesDataset\IKEA\Measurement\test_crop\image2'
save_name = r'IMG_0011_box2.jpg'

if __name__ == '__main__':
    image = cv2.imread(os.path.join(image_dir, image_name))
    roi = cv2.selectROI(windowName="original", img=image, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    print(roi)

    if roi != (0, 0, 0, 0):
        crop = image[y:y + 1024, x:x + 1024]
        cv2.imshow("crop", crop)
        cv2.imwrite(os.path.join(save_dir, save_name), crop)
        print("Saved!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()