"""
程序主框架
"""
from tkinter import ttk

from ..constant import constant
from ..frames import welcome, library, player
from ..common import slots,datahub
from ..constant import constant
from ..data import video

class App:

    def __init__(
        self,
        window,
        w_width,
        w_height,
    ):
        """
        Args:
            window: 程序窗体
            w_width: 应用默认窗体宽度
            w_height: 应用默认窗体高度
        """

        self.window = window

        self.W_WIDTH = w_width
        self.W_HEIGHT = w_height

        self.slots_hub = slots.SlotsHub()
        self.data_hub = datahub.DataHub()

        self.slots_hub.register(constant.SWITCH_FRAME_EVENT, self.switch_window)

        self.cur_display_frame = None

    def get_window(self):
        """
        获取根窗体
        """
        return self.window

    def init(
        self,
        finished_file,
    ):
        """
        对应用进行初始化
        """
        self.__init_app_style()
        self.__init_window()
        self.__load__finished(finished_file)

    def __init_window(self):
        """
        简单初始化窗体
        """
        self.window.title("SoccerDetector")

        # self-adjust: center
        self.window.geometry(str(self.W_WIDTH) + "x" + str(self.W_HEIGHT) + "+" + str((self.window.winfo_screenwidth() - self.W_WIDTH) // 2) + "+" 
        + str((self.window.winfo_screenheight() - self.W_HEIGHT) // 2))

        # init welcome
        self.switch_window(constant.SWITCH_WELCOME_FRAME_CODE)

    def __init_app_style(self):
        """
        对于一些组件进行样式的初始化
        """
        self.shallow_bg_style = ttk.Style()
        self.shallow_bg_style.configure(constant.SHALLOW_FRAME_BACKGROUND_NAME, background = "#6f6f6f")

        self.dark_bg_style = ttk.Style()
        self.dark_bg_style.configure(constant.DARK_FRAME_BACKGROUND_NAME, background = "#3f3f3f")

        self.title_text_style = ttk.Style()
        self.title_text_style.configure(constant.TITLE_TEXT_STYLE_NAME, background = "#6f6f6f", foreground="white",font=('microsoft yahei', 20))
        
        self.separator_line_style = ttk.Style()
        self.separator_line_style.configure(constant.FRAME_SEPARATOR_LINE_NAME, background = "#ffffff")

        self.desc_text_style = ttk.Style()
        self.desc_text_style.configure(constant.DESC_TEXT_STYLE_NAME, background = "#6f6f6f", foreground="white",font=('microsoft yahei', 12))

        self.dark_btn_style = ttk.Style()
        self.dark_btn_style.configure(constant.DARK_BTN_BACKGROUND_NAME, background = "#3f3f3f", foreground = "#3f3f3f", borderwidth = 0)

        self.shallow_btn_style = ttk.Style()
        self.shallow_btn_style.configure(constant.SHALLOW_BTN_BACKGROUND_NAME, background = "#6f6f6f", fg="white",font=('microsoft yahei', 14))
        
    
    def __load__finished(self,file_path):
        """
        加载已经处理完毕的视频标题
        """
        finished_videos = []
        processing_videos = []
        with open(file_path, encoding="utf-8", mode='r') as f:
            for line in f.readlines():
                finished_videos.append(video.Video(name = line[:-1], status = 1))
        self.data_hub.set(constant.FINISHED_VIDEOS, finished_videos)
        self.data_hub.set(constant.PROCESSING_VIDEOS, processing_videos)

    def switch_window(self, window_code, **kwarg):
        """
        根据指定的窗体代码进行页面切换
        Args:
            window_code: 需要切换的目标窗体代码
        """
        if self.cur_display_frame is not None:
            self.cur_display_frame.destory()
        if window_code == constant.SWITCH_WELCOME_FRAME_CODE:
            self.cur_display_frame = welcome.WelcomeFrame(self.window)
        elif window_code == constant.SWITCH_LIBRARY_FRAME_CODE:
            self.cur_display_frame = library.LibraryFrame(self.window)
        elif window_code == constant.SWITCH_PLAYER_FRAME_CODE:
            ...
        else:
            raise NotImplementedError

