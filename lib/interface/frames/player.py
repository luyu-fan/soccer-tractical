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


class ControlButtonCfg:
    """
    对播放面板上一些按钮的封装
    """
    def __init__(self):
        self.show_ball_flag = True
        self.show_kicker_flag = True
        self.play_flag = False
        self.show_bbox_flag = False
        self.show_slow_flag = False
        self.show_vel_flag = False
        self.show_shape_flag = False
        self.show_curve_flag = False
        self.show_tactic_flag = False

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

        self.btn_cfg = ControlButtonCfg()

        self.__init__frame()
    
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
        
        # show ball
        self.ball_display_checkbox_val = tkinter.IntVar()
        self.ball_display_checkbox_val.set(1)
        self.ball_display_checkbox = ttk.Checkbutton(self.control_pannel, text="Ball", variable=self.ball_display_checkbox_val, command=self.show_ball)
        self.ball_display_checkbox.place(relx=0.47, rely=0.25, relwidth=0.05, relheight=0.5)

        # show kicker
        self.kicker_display_checkbox_val = tkinter.IntVar()
        self.kicker_display_checkbox_val.set(1)
        self.kicker_display_checkbox = ttk.Checkbutton(self.control_pannel, text="Kicker", variable=self.kicker_display_checkbox_val, command=self.show_kicker)
        self.kicker_display_checkbox.place(relx=0.51, rely=0.25, relwidth=0.06, relheight=0.5)
        
        # display bbox
        self.bbox_display_check_IntVar = tkinter.IntVar()
        self.bbox_display_check_IntVar.set(0)
        self.bbox_display_checkbox = ttk.Checkbutton(self.control_pannel, text="Bbox", variable=self.bbox_display_check_IntVar, command=self.show_bbox_check)
        self.bbox_display_checkbox.place(relx=0.562, rely=0.25, relwidth=0.05, relheight=0.5)

        # slow down
        self.slow_slow_check_IntVar = tkinter.IntVar()
        self.slow_slow_check_IntVar.set(0)
        self.slow_slow_checkbox = ttk.Checkbutton(self.control_pannel, text="Slowly", variable=self.slow_slow_check_IntVar, command=self.slow_slow_down)
        self.slow_slow_checkbox.place(relx=0.61, rely=0.25, relwidth=0.06, relheight=0.5)

        # show velocity
        self.velocity_checkbox_val = tkinter.IntVar()
        self.velocity_checkbox_val.set(0)
        self.velocity_checkbox = ttk.Checkbutton(self.control_pannel, text="Velocity", variable=self.velocity_checkbox_val, command=self.show_vel)
        self.velocity_checkbox.place(relx=0.661, rely=0.25, relwidth=0.07, relheight=0.5)

        # show team shape
        self.team_shape_checkbox_val = tkinter.IntVar()
        self.team_shape_checkbox_val.set(0)
        self.team_shape_checkbox = ttk.Checkbutton(self.control_pannel, text="Shape", variable=self.team_shape_checkbox_val, command=self.show_shape)
        self.team_shape_checkbox.place(relx=0.72, rely=0.25, relwidth=0.08, relheight=0.5)

        # show distance curve
        self.distance_curve_checkbox_val = tkinter.IntVar()
        self.distance_curve_checkbox_val.set(0)
        self.distance_curve_checkbox = ttk.Checkbutton(self.control_pannel, text="Distance", variable=self.distance_curve_checkbox_val, command=self.show_curve)
        self.distance_curve_checkbox.place(relx=0.77, rely=0.25, relwidth=0.065, relheight=0.5)

        # show tactic
        self.tactic_checkbox_val = tkinter.IntVar()
        self.tactic_checkbox_val.set(0)
        self.tactic_checkbox = ttk.Checkbutton(self.control_pannel, text="Tactics", variable=self.tactic_checkbox_val, command=self.show_tactic)
        self.tactic_checkbox.place(relx=0.832, rely=0.25, relwidth=0.065, relheight=0.5)

        self.back_label = ttk.Button(self.control_pannel, text="返回", command=self.back_library)
        self.back_label.place(relx=0.91, rely=0.25, relwidth=0.08, relheight=0.5)
        # self.back_label.config(style=constant.DESC_TEXT_STYLE_NAME)
        # self.back_label.bind("<Button-1>", self.back_library)

        # 防止关闭窗口产生孤儿线程 即线程泄漏
        self.root.protocol("WM_DELETE_WINDOW", self.destroy_window)

        # ====================================================================================
        # start draw thread
        self.draw_worker.start()

    def get_cur_video(self):
        return self.video

    def get_image_resize(self):
        """
        在绘制时获取展示窗口的像素大小
        """
        width = self.image_label.winfo_width()
        height = self.image_label.winfo_height()
        return (width, height)

    def auto_play(self):
        self.btn_cfg.play_flag = True

    def play_pause(self):
        self.btn_cfg.play_flag = False

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
    
    def show_ball(self):
        """
        显示足球
        """
        self.btn_cfg.show_ball_flag = (self.ball_display_checkbox_val.get() == 1)

    def show_kicker(self):
        """
        显示击球者
        """
        self.btn_cfg.show_kicker_flag = (self.kicker_display_checkbox_val.get() == 1)

    def show_bbox_check(self):
        """
        显示是否显示bbox
        """
        self.btn_cfg.show_bbox_flag = (self.bbox_display_check_IntVar.get() == 1)

    def slow_slow_down(self):
        """
        慢速播放一帧视频
        """
        self.btn_cfg.show_slow_flag = (self.slow_slow_check_IntVar.get() == 1)

    def show_vel(self):
        """
        显式运动速度示例
        """
        self.btn_cfg.show_vel_flag = (self.velocity_checkbox_val.get() == 1)
    
    def show_shape(self):
        """
        显式运动阵型
        """
        self.btn_cfg.show_shape_flag = (self.team_shape_checkbox_val.get() == 1)

    def show_curve(self):
        """
        显示距离曲线
        """
        self.btn_cfg.show_curve_flag = (self.distance_curve_checkbox_val.get() == 1)
    
    def show_tactic(self):
        """
        显式战术
        """
        self.btn_cfg.show_tactic_flag = (self.tactic_checkbox_val.get() == 1)
    
    def back_library(
        self,
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
                frame = cur_video.get_one_rendered_frame(self.btn_cfg)
                if frame is None:
                    continue
                else:
                    # 简单自适应播放窗口大小 (TODO 需要考虑比例情况)
                    frame = cv2.resize(frame, self.get_image_resize(), cv2.INTER_CUBIC)
                    image = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    # avoiding shrinking
                    self.image_label.configure(image=image)
                    self.image_label.image = image
                if self.btn_cfg.show_slow_flag:
                    # for slow slow down: sleeping 100ms
                    time.sleep(0.1)
            except Exception as e:
                pass

        