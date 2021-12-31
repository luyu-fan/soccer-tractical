"""
对视频片段数据及处理的封装
"""
class Video:

    def __init__(
        self,
        name,
        status = 0,
    ):
        """
        Args:
            name: 名称
            status: 状态 可以在初始化时设定 0未完成 1完成
        """
        self.name = name
        self.status = status

    def get_name(self):
        """
        获取资源名称
        """
        return self.name

    def get_cover_img(self):
        """
        获取封面图
        """

    def extract_frames(self):
        """
        将视频抽取为所有的帧
        """