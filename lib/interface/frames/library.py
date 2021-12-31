"""
素材库整理界面，主要包括的功能为展示所有的素材信息，即已经处理分析过的视频和正在后端处理的视频。
Warning:线程不安全的素材库类。
"""
import math
import tkinter, tkinter.ttk as ttk
from PIL import Image, ImageTk

from lib.constant import constant
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
        """
        self.root = root

        # 所处模式
        self.in_finished_mode = True

        # 分页设置
        self.cap_in_page = 8       # 每页最多显示的数目
        self.total_page = 0        # 动态计算的页总数
        self.cur_page = 0          # 当前页
        self.cur_start_index = 0   # 当前页对应的起始索引  
        self.max_cards_num = (self.cap_in_page - self.cap_in_page // 2)                            # 一行中能够显示的最多card数目
        self.card_h_margin = 0.05                                                                  # 横向边距
        self.card_v_margin = 0.1                                                                   # 纵向边距
        self.card_width = (1 - self.card_h_margin * (self.max_cards_num - 1)) / self.max_cards_num
        self.card_height = (1 - self.card_v_margin * 3) / 2

        self.data_hub = datahub.DataHub()
        
        self.videos = []             # 当前用于动态展示的Videos列表
        self.display_cards_list = [] # 展示视频片段的卡片数组

        self.init_frame()
        self.filter_finished()
    
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
        self.right_main_plane.place(relx=0.12, rely=0.05, relwidth=0.85, relheight=0.92)
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
        self.pages_ctl_plane = ttk.Frame(self.right_main_plane)
        self.pages_ctl_plane.place(relx=0.7, rely=0.96, relwidth=0.3, relheight=0.04)
        self.pages_ctl_plane.config(style=constant.SHALLOW_FRAME_BACKGROUND_NAME)
        self.prev_btn = ttk.Label(self.pages_ctl_plane, text="上一页", anchor='center')
        self.prev_btn.place(relx=0.0, rely=0.0, relwidth=0.4, relheight=1.0)
        self.prev_btn.config(style=constant.DESC_TEXT_STYLE_NAME)
        self.pages_info_var = tkinter.StringVar()
        self.page_label = ttk.Label(self.pages_ctl_plane, textvariable = self.pages_info_var, anchor='center')
        self.page_label.place(relx=0.4, rely=0.0, relwidth=0.2, relheight=1.0)
        self.page_label.config(style=constant.DESC_TEXT_STYLE_NAME)
        self.next_btn = ttk.Label(self.pages_ctl_plane, text="下一页", anchor='center')
        self.next_btn.place(relx=0.6, rely=0.0, relwidth=0.4, relheight=1.0)
        self.next_btn.config(style=constant.DESC_TEXT_STYLE_NAME)

        self.prev_btn.bind('<Button-1>', self.prev_page)
        self.next_btn.bind('<Button-1>', self.next_page)

    def prev_page(
        self,
        event
    ):
        """
        Args:
            event: 事件
        """
        self.cur_page -= 1
        self.cur_page = max(self.cur_page, 0)
        self.filter_finished() if self.in_finished_mode else self.filter_processing()

    def next_page(
        self,
        event
    ):
        """
        Args:
            event: 事件
        """
        self.cur_page += 1
        self.filter_finished() if self.in_finished_mode else self.filter_processing()
        
    def filter_finished(self):
        """
        选择处理完毕的视频
        """
        if not self.in_finished_mode:
            self.destory_cards()
            self.cur_page = 0
            self.in_finished_mode = True

        # 动态刷新
        self.videos = self.data_hub.get(constant.FINISHED_VIDEOS)
        total_pages = math.ceil(len(self.videos ) / self.cap_in_page)

        # 循环分页
        if self.cap_in_page * self.cur_page >= len(self.videos): 
            self.cur_page = 0

        # 视频展示
        self.pages_info_var.set(str(self.cur_page + 1) + " / " + str(total_pages) if total_pages > 0 else "- / -")
        self.display_title.set("已处理视频总数: " + str(len(self.videos)))

        self.display()
    
    def filter_processing(self):
        """
        选择处理中的视频
        """
        if self.in_finished_mode:
            self.destory_cards()
            self.cur_page = 0
            self.in_finished_mode = False
        
        # 动态刷新
        self.videos = self.data_hub.get(constant.PROCESSING_VIDEOS)
        total_pages = math.ceil(len(self.videos ) / self.cap_in_page)

        # 循环设置
        if self.cap_in_page * self.cur_page >= len(self.videos): 
            self.cur_page = 0

        # 视频展示
        self.pages_info_var.set(str(self.cur_page + 1) + " / " + str(total_pages) if total_pages > 0 else "- / -")
        self.display_title.set("处理中视频总数: " + str(len(self.videos)))

        self.display()

    def display(self):
        """
        展示所有的视频信息，一页内的内容分上下两排展示。
        """
        # 销毁card
        self.destory_cards()

        # 绘制card
        for i in range((self.cur_page * self.cap_in_page), min((self.cur_page + 1) * self.cap_in_page, len(self.videos))):
            # 分两排绘制
            video = self.videos[i]
            i = i % self.cap_in_page
            if i < self.max_cards_num:
                display_card = card.VideoCard(self.dis_cards_plane, video, (i * (self.card_width + self.card_h_margin), self.card_v_margin), (self.card_width, self.card_height))
            else:
                display_card = card.VideoCard(self.dis_cards_plane, video, ((i - self.max_cards_num) * (self.card_width + self.card_h_margin), self.card_v_margin * 2 + self.card_height), (self.card_width, self.card_height))
            self.display_cards_list.append(display_card)

    def destory_cards(self):
        """
        销毁展示的卡片
        """
        for card in self.display_cards_list:
            card.destory()
        self.display_cards_list = []

    def destory(self):
        
        self.top_level_frame.destroy()


