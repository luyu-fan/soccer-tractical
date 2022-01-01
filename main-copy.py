import threading
import cv2, time, os
import tkinter, tkinter.ttk as ttk

from ttkthemes import ThemedTk
from PIL import Image, ImageTk

from lib.dataprocess import prepare
from lib.render import render
from lib.interaction import interaction
from lib.utils import team_shape
from lib.workthread import thread as wthread

class AnaVideoMeta:
    """
    对视频分析结果的包装
    """
    BUILD_PREPARED = 1           # 应该构建数据
    BUILDING = 2                 # 构建中
    BUILD_ERR = 3                # 构建失败
    FINISHED = 4                 # 数据准备完成

    def __init__(self, videoName):

        self.op_mutex = threading.Lock()

        self.videoName = videoName

        self.KICK_DIST_PIXEL_THRES = 60
        self.SURROUNDING_MAX_DIST_THRES = 400
        self.PROBE_TTL = 60

        self.probe_kicker_cls = ""
        self.probe_kicker_oid = ""
        self.probe_kicker_up_frame_num = 0
        self.frame_num = 1

        self.VIDEO_STATUS = AnaVideoMeta.BUILD_PREPARED
        self.err_info = None

    def set_status(self,status):
        self.op_mutex.acquire(blocking=True)
        self.VIDEO_STATUS = status
        self.op_mutex.release()

    def get_status(self):
        status = None
        self.op_mutex.acquire(blocking=True)
        status = self.VIDEO_STATUS
        self.op_mutex.release()
        return status

    def build(self):
        self.set_status(AnaVideoMeta.BUILDING)
        try:
            self.build_frames()
            self.build_labels()
            self.count_frames()
        except Exception() as e:
            self.err_info = e
            self.set_status(AnaVideoMeta.BUILD_ERR)
        else:
            self.set_status(AnaVideoMeta.FINISHED)

    def build_frames(self):
        prepare.prepare_frames(self.videoName)
    
    def build_labels(self):
        self.labels_dict = prepare.prepare_labels(self.videoName, kick_dist_pixel_thres=self.KICK_DIST_PIXEL_THRES)

    def count_frames(self):
        self.frame_total = 0
        for num in self.labels_dict.keys():
            self.frame_total = max(self.frame_total, int(num))
        return self.frame_total

    def back_n_frames(self, n):
        self.frame_num = max(1, self.frame_num - n)
        self.probe_kicker_cls = ""
        self.probe_kicker_cls = ""
        self.probe_kicker_oid = ""
        self.probe_kicker_up_frame_num = 0

    def next_n_frames(self, n):
        self.frame_num = min(self.frame_total, self.frame_num + n)
        self.probe_kicker_cls = ""
        self.probe_kicker_cls = ""
        self.probe_kicker_oid = ""
        self.probe_kicker_up_frame_num = 0

    # 核心函数
    def get_one_render_frame(self, do_not_incr = False, show_bbox = False):
        if self.frame_num not in self.labels_dict.keys():
            return None
        frame_record = self.labels_dict[self.frame_num]
        img_path = "datasets/images/{:s}/{:06d}.jpg".format(self.videoName.split(".")[0],self.frame_num)
        frame = cv2.imread(img_path)
        ball = frame_record["ball"]
        cur_kicker = frame_record["kicker"]
        if ball is not None:
            # 6 显示bbox
            if show_bbox:
                frame = render.renderBbox_batch(frame, frame_record["bbox"])

            # print("==>", frame_record["ball"].xcenter, frame_record["ball"].ycenter)
            # 1. 将识别到的足球给绘制出来. 标明位置
            frame = render.renderRRectLabel_batch(frame, [ball], color=(36,36,36))
            # 5. 在ttl帧数窗口内探测下一个kicker
            if self.probe_kicker_up_frame_num > self.frame_num:
                for bbox in frame_record["bbox"]:
                    if (self.probe_kicker_oid == bbox.oid and self.probe_kicker_cls == bbox.cls):
                        frame = render.renderRRectLabel_batch(frame, [bbox], color=(0, 0, 255), font_color=(0, 0, 0), label_width=96, label_height=30)
                        break
            else:
                frame_probe_num = self.frame_num + 1
                probe_ttl = self.PROBE_TTL
                while probe_ttl > 0 and (frame_probe_num in self.labels_dict.keys()):
                    probe_kicker = self.labels_dict[frame_probe_num]["kicker"]
                    if probe_kicker is not None and cur_kicker is None:
                        self.probe_kicker_up_frame_num = frame_probe_num
                        self.probe_kicker_cls = probe_kicker.cls
                        self.probe_kicker_oid = probe_kicker.oid
                        break
                    # 这里只是为了修复自我标注时相同的id 如果采用检测的结果就不会有这种问题 此时所有的id都是唯一的
                    if probe_kicker is not None and (probe_kicker.oid != cur_kicker.oid or (probe_kicker.oid == cur_kicker.oid and probe_kicker.cls != cur_kicker.cls)):
                        self.probe_kicker_up_frame_num = frame_probe_num
                        self.probe_kicker_cls = probe_kicker.cls
                        self.probe_kicker_oid = probe_kicker.oid
                        break
                    frame_probe_num += 1
                    probe_ttl -= 1

            if cur_kicker is not None:
                # 3. 将当前帧kicker的周围按照范围将所有的对象检测出来 绘制战术进攻阵型或者防守阵型 显然这里速度很慢 需要批处理 可以看作是一个凸包
                surroundings = interaction.find_surroundings(cur_kicker, frame_record["bbox"], surrounding_max_dist_thres=self.SURROUNDING_MAX_DIST_THRES)

                self_team_shape = team_shape.convexhull_calc(surroundings[0])
                enemy_team_shape = team_shape.convexhull_calc(surroundings[1])
                frame = render.renderTeamShape(frame,self_team_shape,(146,224,186))
                frame = render.renderTeamShape(frame,enemy_team_shape,(224,186,146))
                frame = render.renderRRectLabel_batch(frame, self_team_shape, (242, 168, 123))
                frame = render.renderRRectLabel_batch(frame, enemy_team_shape, (48, 96, 166))
                # 4. 绘制当前kicker到其它队友或者是地方的一个距离 绘制曲线
                frame = render.renderDistance_batch(frame, cur_kicker, self_team_shape, color=(16,255,16))
                frame = render.renderDistance_batch(frame, cur_kicker, enemy_team_shape, color=(16,16,255))

                # 2. 如果当前帧存在kicker 则将当前帧的kicker给绘制出来
                frame = render.renderRRectLabel_batch(frame, [cur_kicker], color=(255, 255, 255), font_color=(0, 0, 0), label_width=96, label_height=30)

        self.frame_num += 1
        if do_not_incr:
            self.frame_num -= 1

        return frame

class SoccerDetector:

    # TODO 动态调整图像 以适应软件变化

    def __init__(self, root):
        
        self.root = root

        self.W_WIDTH = 1280
        self.W_HEIGHT = 780
        self.I_WIDTH = 1280
        self.I_HEIGHT = 720

        self.videos_dict = {}

        video1 = AnaVideoMeta("BXZNP1_17.mp4")
        video2 = AnaVideoMeta("NOSV9C_37.mp4")
        video3 = AnaVideoMeta("BXZNP1_17_Alg.mp4")
        video4 = AnaVideoMeta("Soccer_Long_Demo.mp4")

        self.videos_name_list = ["BXZNP1_17.mp4", "NOSV9C_37.mp4", "BXZNP1_17_Alg.mp4", "Soccer_Long_Demo.mp4"]

        self.videos_dict[video1.videoName] = video1
        self.videos_dict[video2.videoName] = video2
        self.videos_dict[video3.videoName] = video3
        self.videos_dict[video4.videoName] = video4
        
        self.cur_video = None
        # self.cur_video = self.videos_dict[video1.videoName]

        self.draw_worker = wthread.WorkThread("draw_work", self.draw)
        self.cur_video_lock = threading.Lock()

        # Widget
        self.should_auto_play = False
        self.should_display_bbox = False
        self.should_slow_slow_down = False

        # logo
        self.logo_image = Image.open("./assets/logo.jpg")
        self.logo_image = ImageTk.PhotoImage(image = self.logo_image)

        self._exit_signal = False

    def get_cur_video(self):
        self.cur_video_lock.acquire()
        video_obj = self.cur_video
        self.cur_video_lock.release()
        return video_obj

    def exchange_cur_video(self, video_name):
        self.cur_video_lock.acquire()
        self.cur_video = self.videos_dict[video_name]
        self.cur_video_lock.release()

    def init_window(self):

        self.root.title("SoccerDetector")

        # self-adjust: center
        self.root.geometry(str(self.W_WIDTH) + "x" + str(self.W_HEIGHT) + "+" +
                        str((self.root.winfo_screenwidth() - self.W_WIDTH) // 2) + "+" +
                        str((self.root.winfo_screenheight() - self.W_HEIGHT) // 2))

        # draw pannel
        self.draw_pannel = ttk.Frame(self.root) # width=1280, height=720
        self.draw_pannel.place(relx=0, rely=0, relwidth=1.0, relheight=0.92) # pack(side=tkinter.TOP, fill=tkinter.X)
        
        # show
        self.image_label = ttk.Label(self.draw_pannel, image=self.logo_image)
        self.image_label.place(relx=0, rely=0, relwidth=1.0, relheight=1.0) # pack(fill=tkinter.BOTH)

        # init
        # self.image_label.image = self.logo_image

        # control pannel
        self.control_pannel = ttk.Frame(self.root) # width=1280, height=60,
        self.control_pannel.place(relx=0, rely=0.92, relwidth=1.0, relheight=0.08) # .pack(side=tkinter.BOTTOM, fill=tkinter.Y)

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

        # combobox
        self.video_combox_box = ttk.Combobox(self.control_pannel, state = "readonly")
        self.video_combox_box.place(relx=0.89, rely=0.25, relwidth=0.1, relheight=0.5)
        # fixed now TODO add new video
        self.video_combox_box['values'] = self.videos_name_list
        self.video_combox_box.bind("<<ComboboxSelected>>", self.change_video)

        # events
        self.root.protocol("WM_DELETE_WINDOW", self.window_destory)

        # ====================================================================================
        # start draw thread
        self.draw_worker.start()

    def auto_play(self):
        self.should_auto_play = True

    def play_pause(self):
        self.should_auto_play = False

    def back30_frames(self):
        video_obj = self.get_cur_video()
        if video_obj is not None:
            video_obj.back_n_frames(30)

    def back5_frames(self):
        video_obj = self.get_cur_video()
        if video_obj is not None:
            video_obj.back_n_frames(5)

    def next30_frames(self):
        video_obj = self.get_cur_video()
        if video_obj is not None:
            video_obj.next_n_frames(30)

    def next5_frames(self):
        video_obj = self.get_cur_video()
        if video_obj is not None:
            video_obj.next_n_frames(10)
    
    def show_bbox_check(self):
        self.should_display_bbox = (self.bbox_display_check_IntVar.get() == 1)

    def slow_slow_down(self):
        self.should_slow_slow_down = (self.slow_slow_check_IntVar.get() == 1)
        
    def change_video(self, event):
        """
        更新video
        event: 为传进来的事件
        """
        self.exchange_cur_video(self.video_combox_box.get())

    def window_destory(self):
        """
        关闭窗口触发事件
        """
        self._exit_signal = True
        time.sleep(0.3)
        self.root.destroy()
        os._exit(0)

    def draw(self):
        """
        将图像绘制到panel上
        """
        while True:
            if self._exit_signal:
                return
            else:
                cur_video = self.get_cur_video()
                if cur_video is None:
                    continue
                elif cur_video.VIDEO_STATUS != AnaVideoMeta.FINISHED:
                    cur_video.build()
                else:
                    # ≈ 35ms for one frame
                    frame = cur_video.get_one_render_frame(not self.should_auto_play, self.should_display_bbox)
                    if frame is None:
                        continue
                    else:
                        self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        self.image = ImageTk.PhotoImage(image = self.image)
                        # avoiding shrinking
                        self.image_label.configure(image=self.image)
                        self.image_label.image = self.image
                    if self.should_slow_slow_down:
                        # for slow slow down: sleeping 300ms
                        time.sleep(0.3)

def spliting_videos(video_name):
    prepare.prepare_frames(video_name)

if __name__ == "__main__":

    # main_window = tkinter.Tk()
    main_window = ThemedTk(theme="equilux")

    app = SoccerDetector(main_window)
    app.init_window()

    main_window.mainloop()

    # spliting_videos("../")

