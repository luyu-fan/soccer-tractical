import os
import cv2

# 对原始视频进行处理 将其切分为帧
# 其它的一些辅助功能都先不需要设计

from . import get_relative_data_path, check_exists, make_folder

def write_frame(frame, fid, foldler):
    imgPath = os.path.join(foldler, '{:06d}'.format(fid)) + ".jpg"
    cv2.imwrite(imgPath, frame) 

def video2images(videoName):
    """
    将一段视频给切分好若干帧放在文件夹中
    """
    videoPath = os.path.join(get_relative_data_path(), "videos", videoName)
    video = cv2.VideoCapture(videoPath)

    fid = 1
    if video is not None:
        imgsFolder = os.path.join(get_relative_data_path(), "images", videoName.split(".")[0])
        if not check_exists(imgsFolder):
            make_folder(imgsFolder)
        success, frame = video.read()
        while success:
            write_frame(frame, fid, imgsFolder)
            success, frame = video.read()
            fid += 1
    print("Raw Images Prepared!")

