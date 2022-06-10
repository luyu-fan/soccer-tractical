"""
素材实体
"""
from tkinter import ttk
import tkinter
from PIL import ImageTk

from lib.constant import constant
from lib.interface.common import slots
from ..data import video

class VideoCard:
    """
    用于视频信息显示的Card组件
    """
    def __init__(
        self,
        root,
        video,
        pos,
        size,
    ):
        """
        Args:
            root: 父级窗体或控件
            video: 绑定的video的信息
            pos: 相对于父组件布局的位置, 二元组形式(x, y), 绝对比例
            size: 整个卡片组的大小, 二元组形式(width, height), 绝对比例
        """
        self.root = root
        self.video = video
        self.pos = pos
        self.size = size

        self.init_frame()

    def init_frame(self):

        self.card = ttk.Frame(self.root)
        self.card.place(relx=self.pos[0], rely=self.pos[1], relwidth=self.size[0], relheight=self.size[1])
        self.card.config(style=constant.SHALLOW_FRAME_BACKGROUND_NAME)

        # 封面
        self.cover_plane = ttk.Frame(self.card)
        self.cover_plane.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=0.7)
        self.cover_plane.update()   # 先渲染之后才能得到对应的控件像素大小
        self.plane_width = self.cover_plane.winfo_width()
        self.plane_height = self.cover_plane.winfo_height()
        
        # 标题
        if not self.video.is_seg:
            self.title_plane = ttk.Frame(self.card)
            self.title_plane.place(relx=0.0, rely=0.7, relwidth=1.0, relheight=0.15)
            self.title_label = ttk.Label(self.title_plane, text="视频:" + self.video.get_name())
            self.title_label.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
            self.title_label.config(style=constant.DESC_TEXT_STYLE_NAME)

        self.video.cover_update_handler = self.set_cover
        self.video.status_update_handler = self.set_status_info
        
        if self.video.get_status() != video.Video.UNPROCESS:
            self.set_cover()

        # 状态或时长
        if not self.video.is_seg:
            self.video_info_var = tkinter.StringVar(value="时长(帧数): " + str(self.video.get_frames()))
            self.video_info_plane = ttk.Frame(self.card)
            self.video_info_plane.place(relx=0.0, rely=0.85, relwidth=1.0, relheight=0.15)
            self.video_info_label = ttk.Label(self.video_info_plane, textvariable = self.video_info_var)
            self.video_info_label.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
            self.video_info_label.config(style=constant.DESC_TEXT_STYLE_NAME)

    def set_cover(self):
        """
        设置封面
        """
        # 设置视频封面
        self.cover_img = ImageTk.PhotoImage(image = self.video.get_cover_img().resize(size = (self.plane_width, self.plane_height)))
        self.cover_label = ttk.Label(self.cover_plane, image=self.cover_img)
        self.cover_label.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
        self.cover_label.config(style=constant.DESC_TEXT_STYLE_NAME)
        # 绑定双击事件
        self.cover_label.bind("<Double-Button-1>", self.dbclick_play)

    def set_status_info(self, info):
        """
        设置状态信息
        """
        if not self.video.is_seg:
            self.video_info_var.set(info)

    def dbclick_play(
        self,
        event,
    ):
        """
        双击跳转至播放
        """
        if self.video.get_status() == video.Video.FINISHED:
            slots.SlotsHub.get_handler(constant.SWITCH_FRAME_EVENT)(constant.SWITCH_PLAYER_FRAME_CODE, video = self.video)
        else:
            print("请等待处理完毕")

    def hide(self):
        """
        隐藏仅仅销毁View而不销毁数据
        """
        self.card.destroy()

    def destory(self):
        """
        销毁组件中的各个元素
        """
        self.card.destroy()
        self.video.destroy()