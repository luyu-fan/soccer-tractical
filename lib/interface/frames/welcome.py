"""
欢迎界面
"""

from tkinter import ttk
from PIL import Image, ImageTk

from lib.interface.constant import constant

class WelcomeFrame:

    def __init__(
        self,
        app,
    ):
        """
        静态欢迎界面
        Args:
            app: 应用主体
        """
        self.app = app

        self.top_frame = ttk.Frame(app.get_window())
        self.top_frame.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)

        self.logo_image = ImageTk.PhotoImage(image = Image.open("./assets/logo.jpg"))
        self.logo_label = ttk.Label(self.top_frame, image=self.logo_image)
        self.logo_label.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)  # pack(fill=tkinter.BOTH)

        self.enter_btn = ttk.Button(self.top_frame, text = "Enter", command=self.enter_app)
        self.enter_btn.place(relx=0.45, rely=0.7, relwidth=0.1, relheight=0.1)
        
    def enter_app(self):
        self.app.switch_window(constant.LIBRARY_WINDOW)

    def destory(self):
        self.top_frame.destroy()