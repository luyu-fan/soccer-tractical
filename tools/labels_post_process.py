"""
将算法检测出来的结果给整合到一个文件中
1. 先将bbox和id给整合到同一个文件中. frameid, id, cls, x1, y1, w, h
2. 根据上面的结果 为每一个对象划分队伍填充cls字段
"""

import os
import cv2
import numpy as np

# ============================================
# labels processing
# ============================================
# 帧序号从0开始
ball_pos_file = "../datasets/labels/long_demo_ball.txt"
player_pos_file = "../datasets/labels/long_demo_player.txt"

ball_pos_frames_map = {}                                       # 每帧对应的足球位置信息
with open(ball_pos_file, encoding="utf-8", mode="r") as f:
    lines = f.readlines()
    for line in lines:
        line = line[:-1]
        record = line.split(',')
        # print(record)

        frame_id = record[0]
        x1 = int(float(record[1]))
        y1 = int(float(record[2]))
        w = int(float(record[3]) - float(record[1]))
        h = int(float(record[4]) - float(record[2]))

        # 代表是球还是球员
        cls_type = "Ball"

        if frame_id not in ball_pos_frames_map.keys():
            ball_pos_frames_map[frame_id] = []

        ball_pos_frames_map[frame_id].append("Ball,0," + str(x1) + "," + str(y1) + "," + str(w) + "," + str(h))
# print(ball_pos_frames_map)

player_pos_frames_map = {}
with open(player_pos_file, encoding="utf-8", mode="r") as f:
    lines = f.readlines()
    for line in lines:
        line = line[:-1]
        record = line.split(',')
        # print(record)

        frame_id = record[0]
        o_id = record[1]

        x1 = int(float(record[2]))
        y1 = int(float(record[3]))
        w = int(float(record[4]))
        h = int(float(record[5]))

        # 代表是球还是球员
        cls_type = "-"

        if frame_id not in player_pos_frames_map.keys():
            player_pos_frames_map[frame_id] = []

        player_pos_frames_map[frame_id].append(cls_type + "," + o_id + "," + str(x1) + "," + str(y1) + "," + str(w) + "," + str(h))

# print(player_pos_frames_map)

# 将球和球员检测数据合并 形成之前的gt格式用于绘制
merged_pos_result = {}
for frame_id in player_pos_frames_map.keys():
    if frame_id not in merged_pos_result.keys():
        merged_pos_result[frame_id] = []
    if frame_id in ball_pos_frames_map.keys():
        merged_pos_result[frame_id].append(ball_pos_frames_map[frame_id][0])
    merged_pos_result[frame_id].extend(player_pos_frames_map[frame_id])

print(merged_pos_result)


# ============================================
# verify bbox
# ============================================
# imgs_folder = "../datasets/images/BXZNP1_17_Alg/"
# imgs_list = os.listdir(imgs_folder)

# for img_name in imgs_list:
#     img_path = os.path.join(imgs_folder, img_name)
    
#     # print(img_path)
#     frame_id = img_name.split(".")[0].lstrip("0")
#     obj_labels = merged_pos_result[frame_id]
#     img = cv2.imread(img_path)
#     for bbox in obj_labels:
#         bbox = bbox.split(",")
#         x1 = max(0, int(bbox[2]))
#         y1 = max(0, int(bbox[3]))
#         x2 = max(0, int(bbox[2]) + int(bbox[4]))
#         y2 = max(0, int(bbox[3]) + int(bbox[5]))

#         print(x1, y1, x2, y2)

#         if bbox[0] == "Ball":
#             cv2.rectangle(img, (x1, y1), (x2, y2), color = (46,74,244), thickness = 2)
#         else:
#             # print(output[output > 0])
#             # cv2.imshow("output", output)
#             # cv2.waitKey(100)
#             cv2.rectangle(img, (int(bbox[2]), int(bbox[3])), (int(bbox[2]) + int(bbox[4]), int(bbox[3]) + int(bbox[5])), color = (224,213,132), thickness = 1)
#     cv2.imshow("detect", img)
#     cv2.waitKey(100)

# ============================================
# save
# ============================================
save_path = "../datasets/labels/Soccer_Long_Demo_merge.txt"
with open(save_path, encoding="utf-8", mode="w") as f:
    for frame_id in merged_pos_result.keys():
        bboxes = merged_pos_result[frame_id]
        for bbox in bboxes:
            f.write(str(int(frame_id) -1) +"," + bbox + "\n")

cv2.DestroyAllWindows()








