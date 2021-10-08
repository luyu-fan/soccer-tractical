import os,math

from . import check_exists, get_relative_data_path
from .video2frames import check_frames
from utils.distance_calc import calc_distance_in_pixel

class BBoxRecord:
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
    
    def __str__(self) -> str:
        return "[cls:%s, oid:%s, xcenter:%s, ycenter:%s, width:%s, height:%s]".format(str(self.cls), str(self.oid), str(self.xcenter), str(self.ycenter), str(self.width), str(self.height))

def prepare_frames(videoName):
    """
    准备视频原始帧
    """
    videoPath = os.path.join(get_relative_data_path(), "videos", videoName)
    if not check_exists(videoPath):
        raise Exception("Your video file [%s] is not exist in videos folder!".format(videoName))
    else:
        check_frames(videoName)

def prepare_labels(videoName, kick_dist_pixel_thres = 30):
    """
    准备视频对应的跟踪数据标签 即根据标注文件生成每一帧的数据字典
    Args:
        videoName: 视频名称
        kick_dist_pixel_thres: 最小判定像素距离
    """
    # TODO MOT 算法结果
    return prepare_labels_myown(videoName, kick_dist_pixel_thres)

def prepare_labels_myown(videoName, kick_dist_pixel_thres):
    """
    先暂时拿自己已经标注好的视频来做表达最终的表现效果
    文件内容: 单独一个文件，每行记录了这样一个七元组
        frame, cls, id, x1, y1, width, height
        ...
    帧映射内容:
        frames_map: {
            frame_id: {
                "bboxes": [BBoxRecord, .... ....],
                "ball": BBoxRecord,
                "kicker": BBoxRecord,
            }
        }
    Args：
        videoName: 视频名称
        kick_dist_pixel_thres: 判定击球的最小距离
    Return:
        frames_map: dict
    
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
                # 记录当前帧的信息
                frame_id = (int(lineRecord[0]) + 1)
                if  frame_id not in frames_map.keys():
                    frames_map[frame_id] = {}
                    frames_map[frame_id]["bbox"] = []
                    frames_map[frame_id]["ball"] = None
                    frames_map[frame_id]["kicker"] = None
                x1 = int(lineRecord[3])
                y1 = int(lineRecord[4])
                width = int(lineRecord[5])
                height = int(lineRecord[6])
                record = BBoxRecord(lineRecord[1], lineRecord[2], x1 + width / 2, y1 + height / 2, width, height)
                frames_map[frame_id]["bbox"].append(record)
                # 确定是否有ball
                if record.cls == "Ball":
                    frames_map[frame_id]["ball"] = record
    # 将所有的kicker找出来
    for frame_id in frames_map.keys():
        if frames_map[frame_id]["ball"] is not None:
            min_dist, index = math.inf, 0
            ball_center = [frames_map[frame_id]["ball"].xcenter, frames_map[frame_id]["ball"].ycenter]
            for i, bbox in enumerate(frames_map[frame_id]["bbox"]):
                if bbox.cls != "Ball":
                    pixel_dist = calc_distance_in_pixel(ball_center, [bbox.xcenter, bbox.ycenter])
                    if min_dist > pixel_dist:
                        min_dist = pixel_dist
                        index = i
            if min_dist <= kick_dist_pixel_thres:
                frames_map[frame_id]["kicker"] = frames_map[frame_id]["bbox"][index]
    return frames_map
    