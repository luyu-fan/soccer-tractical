"""
对视频片段数据及处理的封装
"""
import os

from PIL import Image

from lib.constant import constant
from lib.dataprocess import check_exists

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

        self.imgs_folder = os.path.join(constant.DATA_ROOT, "images", self.name.split(".")[0])
        self.frames = 0
        self.cover_img = None

    def get_name(self):
        """
        获取资源名称
        """
        return self.name

    def get_cover_img(self):
        """
        获取封面图
        """
        if self.cover_img is not None:
            return self.cover_img
        if check_exists(self.imgs_folder):
            imgs_names = os.listdir(self.imgs_folder)
            if len(imgs_names) > 0:
                img_path = os.path.join(self.imgs_folder, imgs_names[0])
                self.cover_img = Image.open(img_path)
        return self.cover_img

    def get_frames(self):
        """
        返回视频片段的总帧数
        """
        if self.frames != 0: return self.frames
        else:
            if check_exists(self.imgs_folder):
                self.frames = len(os.listdir(self.imgs_folder))
            return self.frames

    def get_status(self):
        """
        获取视频状态
        """
        return self.status

    def extract_frames(self):
        """
        将视频抽取为所有的帧并存放
        """

    def mot_process(self):
        """
        多目标跟踪的处理主流程, 利用多线程处理
        """