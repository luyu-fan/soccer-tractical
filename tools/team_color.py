"""
根据标签中的patch对球队尽心区分
"""

import os
import cv2
import numpy as np

# 1. 先定义几个颜色范围
# 颜色的范围需要根据球队来自定义
color_range_map = {
    'red': (np.asarray([17, 15, 75]), np.asarray([50, 56, 200])),
    'blue': (np.asarray([43, 31, 4]), np.asarray([250, 88, 50])),
    'white': (np.asarray([187, 169, 112]), np.asarray([255, 255, 255])),
    "gray":(np.asarray([0, 0, 46]), np.asarray([180, 43, 220])),
    "black":(np.asarray([0, 0, 0]), np.asarray([180, 255, 46])),
}

imgs_folder = "../datasets/images/Soccer_Long_Demo/"
labels_file = "../datasets/labels/Soccer_Long_Demo_merge.txt"

new_records = []
with open(labels_file, encoding="utf-8", mode="r") as f:
    lines = f.readlines()
    for line in lines:
        bbox = line[:-1].split(",")
        frame_id = int(bbox[0]) + 1
        cls_type = bbox[1]
        is_team_A = False
        if cls_type != "Ball":
            x1 = max(0, int(bbox[3]))
            y1 = max(0, int(bbox[4]))
            x2 = max(0, int(bbox[3]) + int(bbox[5]))
            y2 = max(0, int(bbox[4]) + int(bbox[6]))

            print(x1, y1, x2, y2)

            # TODO optimize reading img
            img_path = imgs_folder + "{:06d}".format(frame_id) + ".jpg"
            print("process:", img_path)
            img = cv2.imread(img_path)

            # 2. 利用掩码计算
            patch = img[y1:y2, x1:x2].copy()
            mask = cv2.inRange(patch, lowerb = color_range_map["white"][0], upperb = color_range_map["white"][1])
            output = cv2.bitwise_and(patch, patch, mask = mask)

            counts = output[:,:,2] > 0
            ratio = counts.sum() / ((y2 - y1) * (x2 - x1))
            is_team_A = ratio > 0.1

        team_flag = "A" if is_team_A else "B"
        new_records.append(line.replace("-,", team_flag+","))

labels_file = "../datasets/labels/Soccer_Long_Demo.txt"
with open(labels_file, encoding="utf-8", mode="w") as f:
    for line in new_records:
        f.write(line)