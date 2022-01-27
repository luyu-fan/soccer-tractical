"""
欢迎界面
"""

import time
from tkinter import ttk
import tkinter
from PIL import Image, ImageTk

from lib.constant import constant
from ..common import slots
class WelcomeFrame:

    def __init__(
        self,
        root,
    ):
        """
        静态欢迎界面
        Args:
            root: 父级窗体或控件
        """
        self.top_frame = ttk.Frame(root)
        self.top_frame.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)
        self.top_frame.config(style=constant.SHALLOW_FRAME_BACKGROUND_NAME)

        self.logo_image = ImageTk.PhotoImage(image = Image.open("./assets/logo.png"))
        self.logo_label = ttk.Label(self.top_frame, image=self.logo_image)
        self.logo_label.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)  # pack(fill=tkinter.BOTH)

        self.enter_btn = tkinter.Button(
            self.top_frame,
            command=self.enter_app,
            font = ('microsoft yahei', 14),
            text = "点击进入",
        )
        self.enter_btn.place(relx=0.45, rely=0.7, relwidth=0.1, relheight=0.08)
        # self.enter_btn.configure(style=constant.WEL_ENTER_BTN)

        self.slots_hub = slots.SlotsHub()
        
    def enter_app(self):
        self.slots_hub.get_handler(constant.SWITCH_FRAME_EVENT)(constant.SWITCH_LIBRARY_FRAME_CODE)

    def destory(self):
        self.top_frame.destroy()
