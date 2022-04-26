"""
程序主框架
"""
import os
from tkinter import ttk

from lib.constant import constant
from lib.dataprocess import check_exists
from ..frames import welcome, library, player, tactics
from ..common import slots, datahub

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

        slots.SlotsHub.register(constant.SWITCH_FRAME_EVENT, self.switch_window)

        self.welcome_frame = None
        self.library_frame = None
        self.player_frame = None
        self.tactic_frame = None

    def get_window(self):
        """
        获取根窗体
        """
        return self.window

    def init(
        self,
    ):
        """
        对应用进行初始化
        """
        self.__init_app_style()
        self.__init_window()
        self.__load__finished()

    def __init_window(self):
        """
        简单初始化窗体
        """
        # self-adjust: center
        self.window.geometry(str(self.W_WIDTH) + "x" + str(self.W_HEIGHT) + "+" + str((self.window.winfo_screenwidth() - self.W_WIDTH) // 2) + "+" 
        + str(int((self.window.winfo_screenheight() - self.W_HEIGHT) // 2.2)))

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

        self.tip_text_style = ttk.Style()
        self.tip_text_style.configure(constant.TIP_TEXT_STYLE_NAME, background = "#3f3f3f", foreground="white",font=('microsoft yahei', 12))

        self.dark_btn_style = ttk.Style()
        self.dark_btn_style.configure(constant.DARK_BTN_BACKGROUND_NAME, background = "#3f3f3f", foreground = "#3f3f3f", borderwidth = 0)

        self.shallow_btn_style = ttk.Style()
        self.shallow_btn_style.configure(constant.SHALLOW_BTN_BACKGROUND_NAME, background = "#6f6f6f", fg="white",font=('microsoft yahei', 14))
        
        self.white_button = ttk.Style()
        self.white_button.configure(constant.WEL_ENTER_BTN, background = "#ffffff", fg="#ffffff",font=('microsoft yahei', 14))
        
    def __load__finished(self):
        """
        加载已经处理完毕的视频标题
        TODO replace by DB or others
        """
        finished_record_file = os.path.join(constant.DATA_ROOT, "record", "finished.txt")
        if check_exists(finished_record_file):
            with open(finished_record_file, encoding="utf-8", mode='r') as f:
                for line in f.readlines():
                    datahub.DataHub.add_finished_video(line[:-1])

    def switch_window(self, window_code, **kwarg):
        """
        根据指定的窗体代码进行页面切换
        Args:
            window_code: 需要切换的目标窗体代码
            **kwarg: 剩余的若干个关键字参数
        """
        # 欢迎界面只会展示一次
        if self.welcome_frame is not None:
            self.welcome_frame.destory()
        if window_code == constant.SWITCH_WELCOME_FRAME_CODE:
            self.welcome_frame = welcome.WelcomeFrame(self.window)
        elif window_code == constant.SWITCH_LIBRARY_FRAME_CODE:
            if self.player_frame is not None:
                self.player_frame.destroy()
                self.player_frame = None
            if self.tactic_frame is not None:
                self.tactic_frame.destory()
                self.tactic_frame = None
            if self.library_frame is None:
                self.library_frame = library.LibraryFrame(self.window)
        elif window_code == constant.SWITCH_PLAYER_FRAME_CODE:
            # 在library_frame上叠加player_frame
            if self.tactic_frame is not None:
                self.tactic_frame.destory()
                self.tactic_frame = None
            if "video" in kwarg.keys():
                self.player_frame = player.PlayerFrame(self.window, kwarg["video"])
            else:
                self.player_frame = player.PlayerFrame(self.window, None)
        elif window_code == constant.SWITCH_TACTICS_FRAME_CODE:
            # 切换到战术信息展示页面 在原始的Library界面上做了一层覆盖
            deafult_video_name = None
            deafult_tactic_type = None
            if "deafult_video_name" in kwarg.keys():
                deafult_video_name = kwarg["deafult_video_name"]
            if "deafult_tactic_type" in kwarg.keys():
                deafult_tactic_type = kwarg["deafult_tactic_type"]
            self.tactic_frame = tactics.TacticsFrame(self.window, deafult_video_name, deafult_tactic_type)
        else:
            raise NotImplementedError

