# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\admin\PycharmProjects\pythonProject3\nerf-paddle")
import cv2
import os
from PIL import Image

images_path = "C:/Users/admin/PycharmProjects/pythonProject3/nerf-paddle/data/nerf_llff_data/data1/images/"
output_dir = "C:/Users/admin/PycharmProjects/pythonProject3/nerf-paddle/data/nerf_llff_data/data1/images_8/"

factor = 8

images_list = os.listdir(images_path)
img = Image.open(images_path + images_list[0])
(W,H) = (img.width,img.height)
print("image_size : ",(W ,H))

for image_name in images_list:
    img = cv2.imread(images_path+image_name)
    img_resize = cv2.resize(img, (int(W/factor), int(H/factor)))
    cv2.imwrite(output_dir + image_name, img_resize)
    print(image_name , " done")
print("all images done")