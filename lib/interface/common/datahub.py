"""
程序运行时涉及到的所有中间数据的集合
"""
from lib.constant import constant
from ..data import video

class DataHub:

    """
    存放和管理程序所需所有数据的静态类
    """
    __global_data_hub = {}

    @staticmethod
    def get(data_name):
        """
        根据数据名称返回对应的数据内容
        Args:
            data_name: 数据名称
        """
        if data_name not in DataHub.__global_data_hub.keys():
            return []
        return DataHub.__global_data_hub[data_name]
    
    @staticmethod
    def set(data_name, data):
        """
        设置对应的数据。当前简单版本会覆盖原来的数据。
        Args:
            data_name: 数据名称
            data: 对应的数据集合
        """
        DataHub.__global_data_hub[data_name] = data
    
    @staticmethod
    def add_processing_video(
        video_path,
    ):
        """
        增加一个待处理的视频到处理队列中
        Args:
            video_path: 视频的绝对路径
        """
        video_name = video_path.split("/")[-1]    # 默认以类Unix下的/作为分隔符
        if constant.PROCESSING_VIDEOS not in DataHub.__global_data_hub:
            DataHub.__global_data_hub[constant.PROCESSING_VIDEOS] = []
        DataHub.__global_data_hub[constant.PROCESSING_VIDEOS].append(video.Video(video_name, status=video.Video.UNPROCESS, upload_video_path = video_path))

    def add_finished_video(
        video_name,
    ):
        """
        增加一个待处理的视频到处理队列中
        Args:
            video_name: 视频名称
        """
        if constant.FINISHED_VIDEOS not in DataHub.__global_data_hub:
            DataHub.__global_data_hub[constant.FINISHED_VIDEOS] = []
        DataHub.__global_data_hub[constant.FINISHED_VIDEOS].append(video.Video(video_name, status=video.Video.LOADED))

    
    