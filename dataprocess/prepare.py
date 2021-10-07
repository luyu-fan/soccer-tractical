import os

from . import check_exists, get_relative_data_path
from .video2frames import check_frames

class LabelRecord:
    """
    代表一个对象的标注框
    """
    def __init__(self, cls, oid, xcenter, ycenter, width, height):
        self.cls = cls
        self.oid = oid
        self.xcenter = xcenter
        self.ycenter = ycenter
        self.width = width
        self.height = height

def prepare_frames(videoName):
    """
    准备视频原始帧
    """
    videoPath = os.path.join(get_relative_data_path(), "videos", videoName)
    if not check_exists(videoPath):
        raise Exception("Your video file [%s] is not exist in videos folder!".format(videoName))
    else:
        check_frames(videoName)

def prepare_labels(videoName):
    """
    准备视频对应的跟踪数据标签 即根据标注文件生成每一帧的数据字典
    """
    # TODO MOT 算法结果
    prepare_labels_myown(videoName)

def prepare_labels_myown(videoName):
    """
    为了演示效果，先暂时拿自己已经标注好的视频来做 先过了演示这一关
    """
    labelFile = videoName.split(".")[0] + ".txt"
    labelPath = os.path.join(get_relative_data_path(), "labels", labelFile)
    frames_map = None
    if not check_exists(labelPath):
        raise Exception("Your label file [%s] is not exist in videos folder!".format(videoName))
    else:
        frames_map = {}
        with open(labelPath, encoding="utf-8", mode="r") as f:
            for line in f.readlines():
                line = line[:-1]
                lineRecord = line.split(",")
                # frame, cls, id, x1, y1, width, height
                if int(lineRecord[0]) not in frames_map.keys():
                    frames_map[int(lineRecord[0])] = []
                x1 = int(lineRecord[3])
                y1 = int(lineRecord[4])
                width = int(lineRecord[5])
                height = int(lineRecord[6])
                record = LabelRecord(lineRecord[1], lineRecord[2], x1 + width / 2, y1 + height / 2, width, height)
                frames_map[int(lineRecord[0])].append(record)
    return frames_map
