"""
程序主框架
"""
from tkinter import ttk
from ..constant import constant
from ..frames import welcome, library, player

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

        self.cur_display_frame = None

    def get_window(self):
        """
        获取根窗体
        """
        return self.window

    def init_window(self):
        """
        简单初始化窗体
        """
        self.window.title("SoccerDetector")

        # self-adjust: center
        self.window.geometry(str(self.W_WIDTH) + "x" + str(self.W_HEIGHT) + "+" + str((self.window.winfo_screenwidth() - self.W_WIDTH) // 2) + "+" 
        + str((self.window.winfo_screenheight() - self.W_HEIGHT) // 2))

        # init welcome
        self.switch_window(constant.WELCOME_WINDOW)

    def switch_window(self, window_code, **kwarg):
        """
        根据指定的窗体代码进行页面切换
        Args:
            window_code: 需要切换的目标窗体代码
        """
        if self.cur_display_frame is not None:
            self.cur_display_frame.destory()
        if window_code == constant.WELCOME_WINDOW:
            self.cur_display_frame = welcome.WelcomeFrame(self)
        elif window_code == constant.LIBRARY_WINDOW:
            print("切换到Library")
            self.cur_display_frame = welcome.WelcomeFrame(self)
        elif window_code == constant.PLAYER_WINDOW:
            ...
        else:
            raise NotImplementedError

