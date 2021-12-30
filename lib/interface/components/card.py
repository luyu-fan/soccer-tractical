"""
素材实体
"""
from tkinter import ttk

from ..constant import constant

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
        
        # TODO 设置图像封面
        self.cover_label = ttk.Label(self.cover_plane, background="#eeffee")
        self.cover_label.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
        self.cover_label.config(style=constant.DESC_TEXT_STYLE_NAME)

        # 标题
        self.title_plane = ttk.Frame(self.card)
        self.title_plane.place(relx=0.0, rely=0.7, relwidth=1.0, relheight=0.15)
        self.title_label = ttk.Label(self.title_plane, text="视频:")
        self.title_label.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
        self.title_label.config(style=constant.DESC_TEXT_STYLE_NAME)

        # 时长
        self.duration_plane = ttk.Frame(self.card)
        self.duration_plane.place(relx=0.0, rely=0.85, relwidth=1.0, relheight=0.15)
        self.duration_label = ttk.Label(self.duration_plane, text="帧数: 123")
        self.duration_label.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
        self.duration_label.config(style=constant.DESC_TEXT_STYLE_NAME)

    def dbclick_play(self):
        """
        双击播放
        """
        ...