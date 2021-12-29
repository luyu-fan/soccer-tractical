"""
素材库整理界面，主要包括的功能为展示所有的素材信息，即已经处理分析过的视频和正在后端处理的视频。
"""

import tkinter, tkinter.ttk as ttk
from PIL import Image, ImageTk

class LibraryFrame:

    def __init__(
        self,
        root,
        global_entities,
    ):
        """
        Args:
            root: 主窗体
            global_entities: 全局视频entities列表
        """
        self.root = root
        self.global_entities = global_entities

        # 根据root窗体构建摆放各个控件的Frame
        self.frame = ttk.Frame(root)

        # 已处理Entities列表
        self.finished_subframe = ttk.Frame(self.frame)
        

        # 处理中Entities列表
        self.processing_subframe = ttk.Frame(self.frame)




