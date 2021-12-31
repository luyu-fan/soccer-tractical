"""
播放器(渲染器界面)
"""
from tkinter import ttk

from lib.constant import constant
from lib.interface.common import slots

class PlayerFrame:

    def __init__(
        self,
        root,
        video,
    ):
        """
        播放器。对每一帧视频进行渲染处理并播放。
        Args:
            root: 父级窗体或控件
            video: 需要播放的视频信息
        """
        self.root = root
        self.video = video

        self.__init__frame()

        self.slots_hub = slots.SlotsHub()

        # TODO 展示播放

    def __init__frame(self):
        """
        绘制
        """
        self.top_level_frame = ttk.Frame(self.root)
        self.top_level_frame.place(relx = 0.0, rely = 0.0, relwidth=1.0, relheight=1.0)
        self.top_level_frame.config(style=constant.DARK_FRAME_BACKGROUND_NAME)

        self.demo_label = ttk.Label(self.top_level_frame, text="DEMO DEMO", anchor="center")
        self.demo_label.place(relx=0.4, rely=0.4, relwidth=0.2, relheight=0.2)
        self.demo_label.config(style=constant.TITLE_TEXT_STYLE_NAME)

        self.back_label = ttk.Label(self.top_level_frame, text="BACK", anchor="center")
        self.back_label.place(relx=0.9, rely=0.9, relwidth=0.1, relheight=0.1)
        self.back_label.config(style=constant.DESC_TEXT_STYLE_NAME)
        self.back_label.bind("<Button-1>", self.back_library)

    def back_library(self, event):
        """
        返回素材库
        """
        self.slots_hub.get_handler(constant.SWITCH_FRAME_EVENT)(constant.SWITCH_LIBRARY_FRAME_CODE)

    def destroy(self):
        """
        销毁和回收所有的控件和线程
        """
        self.top_level_frame.destroy()