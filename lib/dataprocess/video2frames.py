import os
import cv2

# 对原始视频进行处理 将其切分为帧
# 其它的一些辅助功能都先不需要设计

from . import get_relative_data_path, check_exists, make_folder

def write_frame(frame, fid, foldler):
    imgPath = os.path.join(foldler, '{:06d}'.format(fid)) + ".jpg"
    cv2.imwrite(imgPath, frame) 

def video2images(
    videoName,
    video_path = None,
):
    """
    将一段视频给切分好若干帧放在文件夹中
    """
    print("-" * 120)
    print("Raw Images Splitting Start! Default Resize Resolution is 1280 x 720 for a better visualization")
    videoPath = os.path.join(get_relative_data_path(), "videos", videoName) if video_path is None else video_path
    video = cv2.VideoCapture(videoPath)
    fid = 1
    if video is not None:
        imgsFolder = os.path.join(get_relative_data_path(), "images", videoName.split(".")[0])
        if not check_exists(imgsFolder):
            make_folder(imgsFolder)
        success, frame = video.read()
        while success:
            frame = cv2.resize(frame, (1280, 720), cv2.INTER_CUBIC)
            write_frame(frame, fid, imgsFolder)
            success, frame = video.read()
            fid += 1
    video.release()
    print("Raw Images Prepared!")
    print("-" * 120)

def check_frames(
    videoName,
    video_path = None,
):
    """
    根据视频文件名称检查视频对应的帧数 以及视频对应的图片文件夹中的图像数目是否和他对应
    """
    imgsFolder = os.path.join(get_relative_data_path(), "images", videoName.split(".")[0])
    if not check_exists(imgsFolder) or len(os.listdir(imgsFolder)) == 0:
        video2images(videoName, video_path)
    else:
        videoPath = os.path.join(get_relative_data_path(), "videos", videoName) if video_path is None else video_path
        video = cv2.VideoCapture(videoPath)
        totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video.release()
        if len(os.listdir(imgsFolder)) < totalFrames:
            video2images(videoName, video_path)
        else:
            print("Raw Images Has Been Prepared! Nothing to do!")


