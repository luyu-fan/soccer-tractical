import threading
import time
import cv2
import tkinter
from PIL import Image, ImageTk

from lib.dataprocess import prepare
from lib.render import render
from lib.interaction import interaction
from lib.utils import team_shape
from lib.workthread import thread as wthread

class AnaVideoMete:
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

        self.VIDEO_STATUS = AnaVideoMete.BUILD_PREPARED

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
        self.set_status(AnaVideoMete.BUILDING)
        try:
            self.build_frames()
            self.build_labels()
            self.count_frames()
        except Exception() as e:
            self.err_info = e
            self.set_status(AnaVideoMete.BUILD_ERR)
        else:
            self.set_status(AnaVideoMete.FINISHED)

    def build_frames(self):
        prepare.prepare_frames(self.videoName)
    
    def build_labels(self):
        self.labels_dict = prepare.prepare_labels(self.videoName, kick_dist_pixel_thres=self.KICK_DIST_PIXEL_THRES)

    def count_frames(self):
        self.frame_total = 0
        for num in self.labels_dict.keys():
            self.frame_total = max(self.frame_total, int(num))
        return self.frame_total

    def get_one_render_frame(self):
        if self.frame_num not in self.labels_dict.keys():
            return None
        frame_record = self.labels_dict[self.frame_num]
        img_path = "datasets/images/{:s}/{:06d}.jpg".format(self.videoName.split(".")[0],self.frame_num)
        frame = cv2.imread(img_path)
        ball = frame_record["ball"]
        cur_kicker = frame_record["kicker"]
        if ball is not None:
            # print("==>", frame_record["ball"].xcenter, frame_record["ball"].ycenter)
            # 1. 将识别到的足球给绘制出来. 标明位置
            frame = render.renderRRectLabel_batch(frame, [ball], color=(36,36,36))
            if cur_kicker is not None:
                # 3. 将当前帧kicker的周围按照范围将所有的对象检测出来 绘制战术进攻阵型或者防守阵型 显然这里速度很慢 需要批处理 可以看作是一个凸包
                surroundings = interaction.find_surroundings(cur_kicker,frame_record["bbox"], surrounding_max_dist_thres=self.SURROUNDING_MAX_DIST_THRES)
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
                        # 这里只是为了修复自我标注时相同的id 如果采用检测的结果就不会有这种问题 此时所有的id都是唯一的
                        if probe_kicker is not None and (probe_kicker.oid != cur_kicker.oid or (probe_kicker.oid == cur_kicker.oid and probe_kicker.cls != cur_kicker.cls)):
                            self.probe_kicker_up_frame_num = frame_probe_num
                            self.probe_kicker_cls = probe_kicker.cls
                            self.probe_kicker_oid = probe_kicker.oid
                            break
                        frame_probe_num += 1
                        probe_ttl -= 1
        self.frame_num += 1
        return frame

# class FakeAnalyser:

#     def __init__(self):

#         self.video_infos_dict = {}
#         self.cur_video_info = None
#         self.work_thread = wthread.WorkThread("renderring thread", self.rendering)
#         self.render_lock = threading.Lock()

#         # handle hub
#         self.handlers = {
#             AnaVideoMete.BUILD_PREPARED: self.start_build_datasets_handler,
#             AnaVideoMete.BUILDING: self.building_handler,
#             AnaVideoMete.BUILD_ERR: self.build_err_handler,
#             AnaVideoMete.FINISHED: self.finished_handler,
#         }

#         self.render_frame = None

#     def get_cur_video_info(self):
#         """
#         获取当前要处理的视频片段信息
#         """
#         self.render_lock.acquire(blocking=True)
#         video_info = self.cur_video_info
#         self.render_lock.release()
#         return video_info
    
#     def set_cur_video_info(self, videoName):
#         """
#         获取当前要处理的视频片段信息
#         """
#         self.render_lock.acquire(blocking=True)
#         self.cur_video_info = self.video_infos_dict[videoName]
#         self.render_lock.release()
    
#     def add_video_info(self, video_mete):
#         """
#         增加一个新的视频元信息
#         """
#         self.video_infos_dict[video_mete.videoName] = video_mete

#     def rendering(self):
#         """
#         根据video的状态绘制图像帧 线程执行函数
#         """
#         while True:
#             video_mete_obj = self.get_cur_video_info()
#             if video_mete_obj is not None:
#                 self.handlers[video_mete_obj.get_status()]()

#     def start_build_datasets_handler(self):
#         """
#         开始构建数据集
#         """
#         print("start_build_datasets_handler")
#         video_info_obj = self.get_cur_video_info()
#         # in blocking
#         video_info_obj.build()

#     def building_handler(self):
#         """
#         构建中
#         """
#         video_info_obj = self.get_cur_video_info()
#         print(video_info_obj.videoName, "building_handler building")

#     def build_err_handler(self):
#         """
#         构建失败
#         """
#         video_info_obj = self.get_cur_video_info()
#         print(video_info_obj.videoName, "build_err_handler",video_info_obj.err_info)
    
#     def finished_handler(self):
#         """
#         构建成功
#         """
#         video_info_obj = self.get_cur_video_info()
#         # print(video_info_obj.videoName, "finished_handler")
#         self.render_frame = video_info_obj.get_one_render_frame(video_info_obj.frame_num)
#         # print("---",self.render_frame)
    
#     def get_rendered_frame(self):
#         return self.render_frame

#     def start_analysing(self):
#         self.work_thread.start()

#     def stop_analysing(self):
#         self._exit = True



def app(videoName):
    """
    应用入口
    """
    KICK_DIST_PIXEL_THRES = 60
    SURROUNDING_MAX_DIST_THRES = 400
    PROBE_TTL = 60

    prepare.prepare_frames(videoName)
    labels_dict = prepare.prepare_labels(videoName, kick_dist_pixel_thres=KICK_DIST_PIXEL_THRES)

    probe_kicker_cls = ""
    probe_kicker_oid = ""
    probe_kicker_up_frame_num = 0
    frame_num = 1
    while frame_num in labels_dict.keys():
        frame_record = labels_dict[frame_num]
        img_path = "datasets/images/{:s}/{:06d}.jpg".format(videoName.split(".")[0],frame_num)
        frame = cv2.imread(img_path)
        ball = frame_record["ball"]
        cur_kicker = frame_record["kicker"]
        if ball is not None:
            # print("==>", frame_record["ball"].xcenter, frame_record["ball"].ycenter)
            # 1. 将识别到的足球给绘制出来. 标明位置
            frame = render.renderRRectLabel_batch(frame, [ball], color=(36,36,36))
            if cur_kicker is not None:
                # 3. 将当前帧kicker的周围按照范围将所有的对象检测出来 绘制战术进攻阵型或者防守阵型 显然这里速度很慢 需要批处理 可以看作是一个凸包
                surroundings = interaction.find_surroundings(cur_kicker,frame_record["bbox"], surrounding_max_dist_thres=SURROUNDING_MAX_DIST_THRES)
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

                # 5. 在ttl帧数窗口内探测下一个kicker
                if probe_kicker_up_frame_num > frame_num:
                    for bbox in frame_record["bbox"]:
                        if (probe_kicker_oid == bbox.oid and probe_kicker_cls == bbox.cls):
                            frame = render.renderRRectLabel_batch(frame, [bbox], color=(0, 0, 255), font_color=(0, 0, 0), label_width=96, label_height=30)
                            break
                else:
                    frame_probe_num = frame_num + 1
                    probe_ttl = PROBE_TTL
                    while probe_ttl > 0 and (frame_probe_num in labels_dict.keys()):
                        probe_kicker = labels_dict[frame_probe_num]["kicker"]
                        # 这里只是为了修复自我标注时相同的id 如果采用检测的结果就不会有这种问题
                        if probe_kicker is not None and (probe_kicker.oid != cur_kicker.oid or (probe_kicker.oid == cur_kicker.oid and probe_kicker.cls != cur_kicker.cls)):
                            probe_kicker_up_frame_num = frame_probe_num
                            probe_kicker_cls = probe_kicker.cls
                            probe_kicker_oid = probe_kicker.oid
                            break
                        frame_probe_num += 1
                        probe_ttl -= 1
        cv2.imshow("SoccerFrame", frame)
        cv2.waitKey(30)
        frame_num += 1
    cv2.destroyAllWindows()

# class SoccerDetector:

#     def __init__(self, window, analyser):
#         self.window = window
#         self.analyser = analyser
#         self.draw_worker = wthread.WorkThread("draw_work", self.draw)
    
#     def init_window(self):
#         self.window.title("SoccerDetector")
#         self.window.geometry('1280x800+200+100')

#         self.main_frame = tkinter.Frame(self.window)
#         self.main_frame.pack()

#         # pannel
#         self.frame_pannel = tkinter.Frame(self.main_frame, width=1280, height=720)

#         # start draw thread
#         self.draw_worker.start()

#     def draw(self):
#         """
#         将图像绘制到panel上
#         """
#         while True:
#             frame = self.analyser.get_rendered_frame()
#             if frame is None:
#                 continue
#             else:
#                 self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                 self.image.show()
#                 self.image = ImageTk.PhotoImage(image = self.image)
#                 self.image_label = tkinter.Label(self.main_frame, image=self.image)
#                 self.image_label.pack(side=tkinter.TOP)

#     # def render_soccer_frame(self):
#     #     # self.image = cv2.imread(r"""./datasets/images/BXZNP1_17/000001.jpg""")
#     #     self.image = Image.open(r"""./datasets/images/BXZNP1_17/000001.jpg""")
#     #     self.image = ImageTk.PhotoImage(image = self.image)
#     #     self.image_label = tkinter.Label(self.main_frame, image=self.image)
#     #     self.image_label.pack(side=tkinter.TOP)


class SoccerDetector:

    def __init__(self, window):
        self.window = window

        self.videos_dict = {}
        self.cur_video = AnaVideoMete("BXZNP1_17.mp4")

        self.draw_worker = wthread.WorkThread("draw_work", self.draw)
    
    def init_window(self):
        self.window.title("SoccerDetector")
        self.window.geometry('1280x800+200+100')

        self.main_frame = tkinter.Frame(self.window)
        self.main_frame.pack()

        # pannel
        self.frame_pannel = tkinter.Frame(self.main_frame, width=1280, height=720)
        self.frame_pannel.pack()

        # show
        self.image_label = tkinter.Label(self.frame_pannel)
        self.image_label.pack(side=tkinter.TOP)

        # start draw thread
        self.draw_worker.start()

    def draw(self):
        """
        将图像绘制到panel上
        """
        while True:
            if self.cur_video.VIDEO_STATUS != AnaVideoMete.FINISHED:
                self.cur_video.build()
            else:
                frame = self.cur_video.get_one_render_frame()
                if frame is None:
                    continue
                else:
                    self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    # self.image.show()
                    self.image = ImageTk.PhotoImage(image = self.image)
                    self.image_label.configure(image=self.image)
                    self.image_label.image = self.image
                    # if self.image_label is None:
                    #     self.image_label = tkinter.Label(self.main_frame, image=self.image)
                    #     self.image_label.pack(side=tkinter.TOP)
                    # else:
                    #     self.image_label['image'] = self.image
                    # time.sleep(0.1)

    # def render_soccer_frame(self):
    #     # self.image = cv2.imread(r"""./datasets/images/BXZNP1_17/000001.jpg""")
    #     self.image = Image.open(r"""./datasets/images/BXZNP1_17/000001.jpg""")
    #     self.image = ImageTk.PhotoImage(image = self.image)
    #     self.image_label = tkinter.Label(self.main_frame, image=self.image)
    #     self.image_label.pack(side=tkinter.TOP)


if __name__ == "__main__":
    # NOSV9C_37.txt
    # BXZNP1_17.mp4
    # app("NOSV9C_37.mp4")

    # fake_analyser = FakeAnalyser("BXZNP1_17.mp4")
    # fake_analyser.build()

    # frame_total = fake_analyser.count_frames()
    # for frame_num in range(1, frame_total + 1):
    #     frame = fake_analyser.get_one_render_frame(frame_num)
    #     cv2.imshow("SoccerFrame", frame)
    #     cv2.waitKey(30)
    # cv2.destroyAllWindows()

    # main_window = tkinter.Tk()
    # analyser = FakeAnalyser()

    # app = SoccerDetector(main_window, analyser)
    # app.init_window()

    # video1 = AnaVideoMete("BXZNP1_17.mp4")
    # analyser.add_video_info(video1)
    # analyser.set_cur_video_info(video1.videoName)
    # analyser.start_analysing()
    
    # minwindow.render_soccer_frame()
    # fake_analyser = FakeAnalyser(None)
    # video1 = AnaVideoMete("BXZNP1_17.mp4")
    # fake_analyser.add_video_info(video1)
    # fake_analyser.set_cur_video_info("BXZNP1_17.mp4")
    # fake_analyser.start_analyse()


    main_window = tkinter.Tk()

    app = SoccerDetector(main_window)
    app.init_window()

    main_window.mainloop()

