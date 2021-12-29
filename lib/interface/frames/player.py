"""
播放器(渲染器界面)
"""
from tkinter import ttk


class PlayerFrame:

    def __init__(
        self,
        root,
    ):
        """
        播放器。对每一帧视频进行渲染处理并播放。
        Args:
            root: 父容器
        """
        self.root = root

        self.frame = ttk.Frame(self.root)

        # TODO 展示播放