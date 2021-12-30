"""
素材库整理界面，主要包括的功能为展示所有的素材信息，即已经处理分析过的视频和正在后端处理的视频。
"""

import tkinter, tkinter.ttk as ttk
from PIL import Image, ImageTk

from ..constant import constant
from ..components import card
from ..common import datahub

class LibraryFrame:
    
    def __init__(
        self,
        root,
    ):
        """
        Args:
            root: 父级窗体或控件
            global_videos: 全局视频列表
        """
        self.root = root

        # 所处模式
        self.in_finished_mode = True

        # 分页信息
        self.cap_in_page = 8     # 每页最多显示的数目
        self.total_page = 0      # 动态计算的页总数
        self.cur_page = 0        # 当前页
        self.cur_start_index = 0 # 当前页对应的起始索引  
        self.max_cards_num = (self.cap_in_page - self.cap_in_page // 2)                             # 一行中能够显示的最多card数目
        self.card_h_margin = 0.05                                                                  # 横向边距
        self.card_v_margin = 0.1                                                                   # 纵向边距
        self.card_width = (1 - self.card_h_margin * (self.max_cards_num - 1)) / self.max_cards_num
        self.card_height = (1 - self.card_v_margin * 3) / 2

        self.data_hub = datahub.DataHub()

        # 展示视频片段的卡片数组
        self.display_cards_list = []

        self.init_frame()
        self.display()
    
    def init_frame(self):
        """
        frame中各个载体的初始化
        """
        # 根据root窗体构建摆放各个控件的Frame
        self.top_level_frame = ttk.Frame(self.root)
        self.top_level_frame.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)
        self.top_level_frame.config(style=constant.SHALLOW_FRAME_BACKGROUND_NAME)

        # left_bar 左侧选项菜单
        self.left_bar = ttk.Frame(self.top_level_frame)
        self.left_bar.place(relx=0, rely=0, relwidth=0.1, relheight=1.0)
        self.left_bar.config(style=constant.DARK_FRAME_BACKGROUND_NAME)

        # finished 菜单按钮
        self.finished_image = ImageTk.PhotoImage(image = Image.open("./assets/finished.png"))
        self.finished_btn = ttk.Button(self.left_bar,image=self.finished_image, command=self.filter_finished)
        self.finished_btn.place(relx=0, rely=0, relwidth=1, relheight=0.15)
        self.finished_btn.config(style=constant.DARK_BTN_BACKGROUND_NAME)

        self.proc_image = ImageTk.PhotoImage(image = Image.open("./assets/processing.png"))
        self.proc_btn = ttk.Button(self.left_bar,image=self.proc_image, command=self.filter_processing)
        self.proc_btn.place(relx=0, rely=0.15, relwidth=1, relheight=0.15)
        self.proc_btn.config(style=constant.DARK_BTN_BACKGROUND_NAME)

        # right_main_plane 右侧显示主体
        self.right_main_plane = ttk.Frame(self.top_level_frame)
        self.right_main_plane.place(relx=0.12, rely=0.05, relwidth=0.85, relheight=0.9)
        self.right_main_plane.config(style=constant.SHALLOW_FRAME_BACKGROUND_NAME)

        # dis_title_plane 统计信息
        self.dis_title_plane = ttk.Frame(self.right_main_plane)
        self.dis_title_plane.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=0.1)
        self.display_title = tkinter.StringVar()
        self.title_label = ttk.Label(self.dis_title_plane, textvariable=self.display_title)
        self.title_label.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
        self.title_label.config(style=constant.TITLE_TEXT_STYLE_NAME)
        
        self.separator = ttk.Frame(self.dis_title_plane)
        self.separator.place(relx=0.0, rely=0.98, relwidth=1.0, relheight=0.01)
        self.separator.config(style=constant.FRAME_SEPARATOR_LINE_NAME)

        # dis_cards_plane 各个卡片显示信息
        self.dis_cards_plane = ttk.Frame(self.right_main_plane)
        self.dis_cards_plane.place(relx=0.0, rely=0.1, relwidth=1.0, relheight=0.9)
        self.dis_cards_plane.config(style=constant.SHALLOW_FRAME_BACKGROUND_NAME)

        # next prev btn 上一页下一页的按钮
        # TODO 按钮实现

    def filter_finished(self):
        """
        选择处理完毕的视频
        """
        self.in_finished_mode = True
        self.display_title.set("已处理视频总数:" + str(len(self.data_hub.get(constant.FINISHED_VIDEOS))))
        print("Select Finished")
    
    def filter_processing(self):
        """
        选择处理中的视频
        """
        self.in_finished_mode = False
        self.display_title.set("正在处理:" + str(len(self.data_hub.get(constant.FINISHED_VIDEOS))))
        print("Select Processing")

    def display(self):
        """
        展示所有的视频信息，一页内的内容分上下两排展示。
        """
        # 绘制已经处理完毕可播放的
        # videos = self.processed_videos_list[self.cur_start_index:self.cur_start_index+self.cap_in_page]
        # for (i, video) in enumerate(videos):
        videos = self.data_hub.get(constant.FINISHED_VIDEOS) if self.in_finished_mode else self.data_hub.get(constant.PROCESSING_VIDEOS)
        if self.cap_in_page * self.cur_page >= len(videos):
            self.cur_page = 0
        # 更新统计信息
        for i in range((self.cur_page * self.cap_in_page), min((self.cur_page + 1) * self.cap_in_page, len(videos))):
            # 分两排绘制
            i = i % self.cap_in_page
            if i < self.max_cards_num:
                display_card = card.VideoCard(self.dis_cards_plane, None, (i * (self.card_width + self.card_h_margin), self.card_v_margin), (self.card_width, self.card_height))
            else:
                display_card = card.VideoCard(self.dis_cards_plane, None, ((i - self.max_cards_num) * (self.card_width + self.card_h_margin), self.card_v_margin * 2 + self.card_height), (self.card_width, self.card_height))
            self.display_cards_list.append(display_card)

    def destory(self):
        
        self.top_level_frame.destroy()


