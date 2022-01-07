"""
播放器(渲染器界面)
"""
import threading, tkinter, time, os
import cv2

from PIL import Image, ImageTk
from tkinter import ttk

from lib.constant import constant
from lib.interface.common import slots
from lib.interface.data.video import Video
from lib.workthread import thread as wthread

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

        # 播放相关
        self.draw_worker = wthread.WorkThread("draw_work", self.draw)
        self.cur_video_lock = threading.Lock()

        # Widget
        self.should_auto_play = False
        self.should_display_bbox = False
        self.should_slow_slow_down = False

        self.__init__frame()

    def get_cur_video(self):
        return self.video

    def __init__frame(self):
        """
        绘制
        """
        self.top_level_frame = ttk.Frame(self.root)
        self.top_level_frame.place(relx = 0.0, rely = 0.0, relwidth=1.0, relheight=1.0)
        self.top_level_frame.config(style=constant.DARK_FRAME_BACKGROUND_NAME)

        # draw pannel
        self.draw_pannel = ttk.Frame(self.top_level_frame)
        self.draw_pannel.place(relx=0, rely=0, relwidth=1.0, relheight=0.92)
        
        # show
        self.image_label = ttk.Label(self.draw_pannel)
        self.image_label.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)

        # control pannel
        self.control_pannel = ttk.Frame(self.top_level_frame)
        self.control_pannel.place(relx=0, rely=0.92, relwidth=1.0, relheight=0.08)

        # buttons
        self.auto_play_btn = ttk.Button(self.control_pannel, text="播放", command=self.auto_play)
        self.pause_btn = ttk.Button(self.control_pannel,text="暂停",command=self.play_pause)
        self.auto_play_btn.place(relx=0.01, rely=0.25, relwidth=0.05, relheight=0.5)
        self.pause_btn.place(relx=0.06, rely=0.25, relwidth=0.05, relheight=0.5)

        self.pre_10_frames = ttk.Button(self.control_pannel, text="<< 后退30帧", command=self.back30_frames)
        self.pre_1_frames = ttk.Button(self.control_pannel,text="< 后退5帧", command=self.back5_frames)
        self.next_1_frames = ttk.Button(self.control_pannel,text="前进5帧 >", command=self.next5_frames)
        self.next_10_frames = ttk.Button(self.control_pannel, text="前进30帧 >>", command=self.next30_frames)
        self.pre_10_frames.place(relx=0.15, rely=0.25, relwidth=0.08, relheight=0.5)
        self.pre_1_frames.place(relx=0.23, rely=0.25, relwidth=0.08, relheight=0.5)
        self.next_1_frames.place(relx=0.31, rely=0.25, relwidth=0.08, relheight=0.5)
        self.next_10_frames.place(relx=0.39, rely=0.25, relwidth=0.08, relheight=0.5)
        
        # display bbox
        self.bbox_display_check_IntVar = tkinter.IntVar()
        self.bbox_display_check_IntVar.set(0)
        self.bbox_display_checkbox = ttk.Checkbutton(self.control_pannel, text="Bbox", variable=self.bbox_display_check_IntVar, command=self.show_bbox_check)
        self.bbox_display_checkbox.place(relx=0.48, rely=0.25, relwidth=0.05, relheight=0.5)

        # slow slow down
        self.slow_slow_check_IntVar = tkinter.IntVar()
        self.slow_slow_check_IntVar.set(0)
        self.slow_slow_checkbox = ttk.Checkbutton(self.control_pannel, text="Slowly", variable=self.slow_slow_check_IntVar, command=self.slow_slow_down)
        self.slow_slow_checkbox.place(relx=0.53, rely=0.25, relwidth=0.06, relheight=0.5)

        self.back_label = ttk.Label(self.top_level_frame, text="BACK", anchor="center")
        self.back_label.place(relx=0.9, rely=0.9, relwidth=0.1, relheight=0.1)
        self.back_label.config(style=constant.DESC_TEXT_STYLE_NAME)
        self.back_label.bind("<Button-1>", self.back_library)

        # 防止关闭窗口产生孤儿线程 即线程泄漏
        self.root.protocol("WM_DELETE_WINDOW", self.destroy_window)

        # ====================================================================================
        # start draw thread
        self.draw_worker.start()

    def auto_play(self):
        self.should_auto_play = True

    def play_pause(self):
        self.should_auto_play = False

    def back30_frames(self):
        video = self.get_cur_video()
        if video is not None:
            video.back_n_frames(30)

    def back5_frames(self):
        video = self.get_cur_video()
        if video is not None:
            video.back_n_frames(5)

    def next30_frames(self):
        video = self.get_cur_video()
        if video is not None:
            video.next_n_frames(30)

    def next5_frames(self):
        video = self.get_cur_video()
        if video is not None:
            video.next_n_frames(10)
    
    def show_bbox_check(self):
        """
        显示是否显示bbox
        """
        self.should_display_bbox = (self.bbox_display_check_IntVar.get() == 1)

    def slow_slow_down(self):
        """
        慢速播放一帧视频
        """
        self.should_slow_slow_down = (self.slow_slow_check_IntVar.get() == 1)

    def back_library(
        self, 
        event
    ):
        """
        返回素材库
        """
        self.destroy()
        slots.SlotsHub.get_handler(constant.SWITCH_FRAME_EVENT)(constant.SWITCH_LIBRARY_FRAME_CODE)

    def destroy(self):
        """
        销毁和回收所有的控件和线程
        """
        if self.draw_worker.is_alive() and not self.draw_worker.is_stop():
            self.draw_worker.stop()
        self.top_level_frame.destroy()

    def destroy_window(self):
        self.destroy()
        os._exit(0)

    def draw(self, stop_event):
        """
        将图像绘制到panel上 逐帧绘制的关键函数
        """
        cur_video = self.get_cur_video()
        while cur_video is not None:
            if stop_event.is_set():
                return
            try:
                # ≈ 35ms for one frame
                frame = cur_video.get_one_rendered_frame(not self.should_auto_play, self.should_display_bbox)
                if frame is None:
                    continue
                else:
                    image = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    # avoiding shrinking
                    self.image_label.configure(image=image)
                    self.image_label.image = image
                if self.should_slow_slow_down:
                    # for slow slow down: sleeping 300ms
                    time.sleep(0.3)
            except Exception as e:
                pass

        