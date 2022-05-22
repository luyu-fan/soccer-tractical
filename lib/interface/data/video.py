"""
对视频片段数据及处理的封装
"""
import os
import math
import threading
import cv2

from PIL import Image
from sklearn.cluster import KMeans

from lib.constant import constant
from lib.dataprocess import check_exists
from lib.dataprocess import prepare
from lib.interface.data.tacticfsm import TacticFSM
from lib.render import render
from lib.interaction import interaction
from lib.utils import team_shape
from lib.workthread import thread as wthread
from lib.coloring import rgb2cn
from lib.interface.data.postfixfsm import *

from lib.interface.common import datahub
from lib.net import client
       
class Video:

    LOADED = 0                     # 可直接播放的分析视频已加载
    UNPROCESS = 1                  # 数据未处理
    EXTRACT = 2                    # 抽帧
    TRACKING = 3                   # 跟踪
    INTERMEDIATE = 4               # 中间数据处理
    FINISHED = 5                   # 数据已经过服务器处理

    INFO = 0
    DEBUG = 1
    RELEASE = 2
    ERROR = 3
    FATAL = 4

    LOG_STRS_MAP = {
        INFO: "INFO",
        DEBUG: "DEBUG",
        RELEASE: "RELEASE",
        ERROR: "ERROR",
        FATAL: "FATAL"
    }

    def log(self, level, desc):
        """
        将video处理过程中的状态日志进行输出
        TODO 有时间将日志部分模块化处理 规范功能和代码
        """
        if level >= self.log_level:
            print("[%s]: %s" % (Video.LOG_STRS_MAP[level], desc))

    def __init__(
        self,
        name,
        status = 1,
        upload_video_path = None,
        is_seg = False,
        tactic_type = None,
    ):
        """
        Args:
            name: 名称
            status: 状态 可以在初始化时设定 0未完成 1完成 2中间数据处理
            upload_video_path: 对应的原始上传的视频文件路径
            is_seg: 是否是一个对应了tactic的segment
            tactic_type: 在表示为一个seg的情况下代表了战术类型
        """
        # 互斥锁
        self.op_mutex = threading.Lock()

        # 基础信息
        self.name = name
        self.video_status = status
        self.upload_video_path = upload_video_path
        self.imgs_folder_name = os.path.join(constant.DATA_ROOT, "images", self.name.split(".")[0])
        self.labels_folder_name = os.path.join(constant.DATA_ROOT, "labels")
        self.total_frames = 0
        self.cover_img = None
        self.is_seg = is_seg
        self.tactic_type = tactic_type

        # 和视频播放相关的控制
        # TODO 作为可以调节的参数暴露在GUI上
        self.KICK_DIST_PIXEL_THRES = 60
        self.SURROUNDING_MAX_DIST_THRES = 300
        self.PROBE_TTL = 45

        # 视频序列往前探测分析对象时所需要做的标记
        self.probe_kicker_cls = ""
        self.probe_kicker_oid = ""
        self.probe_kicker_up_frame_num = 0
        self.cur_frame_num = 1
        
        # TacticFSM
        self.tactic_fsm = TacticFSM()

        # 在战术绘制时不合法的帧(过短导致)
        self.illegal_tactics_frames = set([])
        # 在使用战术分析的时候用来判断是否已经播放结束
        self.start_frame_num = 1
        self.end_frame_num = -1

        # 空间状态更新句柄
        self.cover_update_handler = None
        self.status_update_handler = None

        self.process_thread = None
        if not self.is_seg:
            # 网络客户端 (将视频处理任务看作是互相隔离的客户端)
            self.client = client.MOTClient(constant.REMOTE_IP, constant.REMOTE_PORT) if self.video_status == Video.UNPROCESS else None

            # 数据处理线程
            self.process_thread = wthread.WorkThread("video_process:"+self.name, self.process)
            self.process_thread.start()

        # 处理结果列表
        self.players_list = []
        self.ball_list = []

        # 日志映射 TODO 增加日志模块 代理全局日志 规范化处理
        self.log_level = Video.RELEASE
    
    def process(self, stop_event):
        """
        在视频展示之间完成一些准备工作。
        已处理完成视频: 根据跟踪结果完成绘制所需要数据的生成
        新上传视频片段: 跟踪以及后续处理
        Args:
            stop_event: 退出事件信号
        """
        if self.get_status() == Video.LOADED:
            self.process_loaded(stop_event)   # 得到所有的标签
            # self.extract_tactics_segments()   # 抽取出所有的战术片段
            self.extract_tactics_by_fsm()
        else:
            self.process_unprocess(stop_event)

    def process_loaded(self, stop_event):
        """
        已跟踪处理过的视频
        """
        if stop_event.is_set():
            return
        self.build_labels()
        self.set_status(Video.FINISHED)

    def process_unprocess(self, stop_event):
        """
        处理新上传视频
        流程:
            1. 视频抽帧
            2. 每帧视频服务器处理
            3. 中间结果整合
            4. 写入文件
            5. 执行分队算法
            6. 处理完成 切换到处理完毕集合可以渲染 (TODO 有时间将处理完毕的结果同时写入文件或者数据库)
        """
        # 1.视频抽帧
        self.extract_frames()
        if stop_event.is_set(): return
        
        # 2. 每帧视频服务器处理
        self.main_track(stop_event)
        if stop_event.is_set(): return

        # 3. 中间数据后处理
        try:
            if self.status_update_handler is not None:  self.status_update_handler("状态: 中间数据处理")
        except Exception as e:
            ...
        self.set_status(Video.INTERMEDIATE)
        self.process_data()

        # 4. 自动颜色分队算法
        self.coloring(stop_event)
        if stop_event.is_set(): return

        # 5. 写入label文件
        self.save_labels()
        try:
            if self.status_update_handler is not None:  self.status_update_handler("状态: 处理完毕")
        except Exception as e:
            ...

        # 6. 移动到不同的队列中
        self.set_status(Video.FINISHED)
        self.move_to_loaded()
    
    def extract_frames(self):
        """
        将视频转化为帧序列
        """
        # 1. 抽帧
        self.set_status(Video.EXTRACT)
        prepare.prepare_frames(self.name, video_path = self.upload_video_path)
        # 无关紧要的状态更新使用异常机制包裹起来避免异常
        try:
            if self.cover_update_handler is not None: self.cover_update_handler()
        except Exception as e:
            ...

    def main_track(self, stop_event):
        """
        跟踪主流程
        """
        # 2. 视频跟踪处理
        try:
            self.client.connect()
        except Exception as e:
            self.client.close()
            return

        try:
            if self.status_update_handler is not None: self.status_update_handler("状态: 跟踪处理 0%")
        except Exception as e:
            ...

        self.set_status(Video.TRACKING)
        # 使用Client进行跟踪处理
        for (i, image_name) in enumerate(os.listdir(self.imgs_folder_name)):
            
            # 检查退出
            if stop_event.is_set():
                self.client.close()
                return
                
            img_path = os.path.join(self.imgs_folder_name, image_name)
            frame = cv2.imread(img_path)
            self.client.send(i + 1, frame)
            result = self.client.recv()

            frame_result = []
            player_result = result['player']
            for bbox, oid in zip(player_result[0], player_result[1]):
                x1y1wh_box = ["", oid, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                frame_result.append(x1y1wh_box)
                # cv2.rectangle(frame, (x1y1wh_box[2], x1y1wh_box[3]), (x1y1wh_box[2] + x1y1wh_box[4], x1y1wh_box[3] + x1y1wh_box[5]), color=(23,45,67), thickness=2)
            self.players_list.append(frame_result)

            frame_result = []
            ball_result = result["ball"]
            for bbox, oid in zip(ball_result[0], ball_result[1]):
                print(bbox)
                x1y1wh_box = ["Ball", oid, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                frame_result.append(x1y1wh_box)
                # cv2.rectangle(frame, (x1y1wh_box[2], x1y1wh_box[3]), (x1y1wh_box[2] + x1y1wh_box[4], x1y1wh_box[3] + x1y1wh_box[5]), color=(12,45,240), thickness=2)
            if len(frame_result) == 0:
                frame_result.append(None)
            # 仅仅使用一个检测结果作为当前帧的代表
            self.ball_list.append(frame_result[0])

            # cv2.imshow("frame", frame)
            # cv2.waitKey(2000)
            # cv2.destroyAllWindows()

            try:
                if self.status_update_handler is not None: self.status_update_handler("状态: 跟踪处理 %d%%" % (int((i + 1) / self.get_frames() * 100), ))
            except Exception as e:
                ...

        self.client.close()
    
    def save_labels(self):
        """
        保持处理好的标注, 下次直接使用
        """
        if not check_exists(self.labels_folder_name):
            os.mkdir(self.labels_folder_name)
        
        label_file = os.path.join(self.labels_folder_name, self.name.split(".")[0] + ".txt")
        with open(label_file, encoding="utf-8", mode="w") as f:
            for (frame_id, frame_mot_result) in enumerate(self.players_list):
                # players
                for i in range(len(frame_mot_result)):
                    # frame_id,cls,oid,x1,y1,w,h
                    record = str(frame_id) + "," + \
                            frame_mot_result[i][0] +"," + \
                            str(frame_mot_result[i][1]) +"," + \
                            str(frame_mot_result[i][2]) +"," + \
                            str(frame_mot_result[i][3]) +"," + \
                            str(frame_mot_result[i][4]) +"," + \
                            str(frame_mot_result[i][5]) + "\n"
                    f.write(record)
                # ball
                if self.ball_list[frame_id] is not None:
                    record = str(frame_id) + "," + \
                            self.ball_list[frame_id][0] +"," + \
                            str(self.ball_list[frame_id][1]) +"," + \
                            str(self.ball_list[frame_id][2]) +"," + \
                            str(self.ball_list[frame_id][3]) +"," + \
                            str(self.ball_list[frame_id][4]) +"," + \
                            str(self.ball_list[frame_id][5]) + "\n"
                    f.write(record)
                    
    def process_data(self):
        """
        利用有限状态机对足球轨迹进行修正
        """
        # result = {
        #     "player": self.players_list,
        #     "ball": self.ball_list,
        # }
        # with open("result.pkl", mode="wb") as f:
        #     pickle.dump(result, f)
        # print("服务器处理结果已写入文件")

        # with open("result.pkl", mode="rb") as f:
        #     result = pickle.load(f)

        # self.ball_list = result["ball"]
        # self.players_list = result['player']
        
        fsm = PostFixFSM(
            window_size=10,
            activate_thres=0.6,
            frames_result=self.ball_list,
        )
        fsm.run()

    def move_to_loaded(self):
        """
        移动至已处理队列
        """
        datahub.DataHub.move(self)
    
    def coloring(self, stop_event):
        """
        根据CN特征对球员进行分队{A, B, C}
        步骤:
        1. 获取每个目标的bbox区域内的图像
        2. 获取目标图像每个像素值对应的CN
        3. 将整幅图像所有CN的均值作为当前图像的CN代表
        4. 利用K-means算法对所有目标的CN代表进行聚类，得到两个中心点(如果包括教练之类的外围目标的话那就是三个)
        5. 根据每个点到聚类中心的距离为每个目标分配类别
        """
        # 目标的CN代表 二维矩阵 obj_num * cn_dim
        obj_cn_reps = []
        for frame_mot_result, image_name in zip(self.players_list, os.listdir(self.imgs_folder_name)):
            # 加载图像
            img_path = os.path.join(self.imgs_folder_name, image_name)
            frame = cv2.imread(img_path)
            for i in range(len(frame_mot_result)):
                if stop_event.is_set(): return
                x1 = max(0, frame_mot_result[i][2])
                y1 = max(0, frame_mot_result[i][3])
                x2 = x1 + frame_mot_result[i][4]
                y2 = y1 + frame_mot_result[i][5]
                obj_cn_rep = rgb2cn.get_img_mean_rep(frame[y1:y2, x1:x2, :].copy(), remove_green=True)
                obj_cn_reps.append(obj_cn_rep)

        # TODO 选择全部的样本非常慢 实际上可以考虑采用部分目标进行K-Means聚类 剩余的部分仅仅根据聚类结果参与分配
        # 先利用前若干帧中的对象形成的CN特征进行K-Means聚类，然后后续所有的对象目标计算与这个颜色中心的距离以此作为自己的队伍颜色划分
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(obj_cn_reps)

        # print(kmeans.labels_)
        # print(kmeans.cluster_centers_)

        # 根据具体数量来划分ABC
        counters = {
            0: 0,
            1: 0,
            2: 0
        }
        for l in kmeans.labels_:
            counters[l] += 1
        
        sorted_keys = sorted(counters, key= lambda x: -counters[x])
        # print(counters, sorted_keys)

        index = 0
        for frame_mot_result in self.players_list:
            for i in range(len(frame_mot_result)):
                if stop_event.is_set(): return
                if kmeans.labels_[index] == sorted_keys[0]:
                    # print("A")
                    frame_mot_result[i][0] = "A"   # 分配给A队
                elif kmeans.labels_[index] == sorted_keys[1]:
                    frame_mot_result[i][0] = "B"   # 分配给B队
                    # print("B")
                else:
                    frame_mot_result[i][0] = "J"   # 分配给J队
                index += 1
            
    def get_name(self):
        """
        获取资源名称
        """
        return self.name

    def get_cover_img(self):
        """
        获取封面图
        """
        if self.cover_img is not None:
            return self.cover_img
        if check_exists(self.imgs_folder_name):
            cover_img_name = "%06d.jpg" %(self.cur_frame_num,)
            img_path = os.path.join(self.imgs_folder_name, cover_img_name)
            self.cover_img = Image.open(img_path)
        return self.cover_img

    def get_frames(self):
        """
        返回视频片段的总帧数
        """
        if self.is_seg:
            self.total_frames = self.end_frame_num - self.start_frame_num + 1
        elif check_exists(self.imgs_folder_name):
            self.total_frames = len(os.listdir(self.imgs_folder_name))
            self.end_frame_num = self.total_frames
        return self.total_frames

    def set_status(
        self,
        status
    ):
        """
        设置状态
        """
        self.op_mutex.acquire(blocking=True)
        self.video_status = status
        self.op_mutex.release()

    def get_status(self):
        status = None
        self.op_mutex.acquire(blocking=True)
        status = self.video_status
        self.op_mutex.release()
        return status

    def build_frames(self):
        """
        将视频抽取为所有的图像帧并存放
        """
        prepare.prepare_frames(self.name)

    def build_labels(self):
        """
        根据当前的播放的设置加载渲染所需的标注数据
        """
        self.labels_dict = prepare.prepare_labels(self.name, kick_dist_pixel_thres=self.KICK_DIST_PIXEL_THRES)

    def back_n_frames(
        self,
        n
    ):
        """
        回退n帧
        """
        self.cur_frame_num = max(self.start_frame_num, self.cur_frame_num - n)
        self.probe_kicker_cls = ""
        self.probe_kicker_cls = ""
        self.probe_kicker_oid = ""
        self.probe_kicker_up_frame_num = 0

    def next_n_frames(
        self, 
        n
    ):
        """
        前进若干帧
        """
        self.cur_frame_num = min(self.end_frame_num, self.cur_frame_num + n)
        self.probe_kicker_cls = ""
        self.probe_kicker_cls = ""
        self.probe_kicker_oid = ""
        self.probe_kicker_up_frame_num = 0

    def __load_frame(self):
        """
        根据图像路径读取图像数据
        """
        img_path = os.path.join(constant.DATA_ROOT, "images", self.name.split(".")[0], "{:06d}.jpg".format(self.cur_frame_num))
        frame = cv2.imread(img_path)

        return frame

    def __load_frame_data(self):
        """
        加载绘制时的每一帧标签数据
        """
        frame_record = self.labels_dict[self.cur_frame_num]
        self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " data prepared.")

        return frame_record

    def __render_object_bbox(
        self,
        frame,
        frame_record
    ):
        """
        绘制每一帧中所有的对象的bbox
        """
        frame = render.renderBbox_batch(frame, frame_record["bbox"])
        self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " render bbox finished.")
        return frame

    def __render_ball(
        self,
        frame,
        ball
    ):
        """'
        绘制球信息
        """
        frame = render.renderRRectLabel_batch(frame, [ball], color=(36,36,36))
        self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " render rect and label finished.")
        return frame

    def __find_next_kicker(
        self,
        cur_kicker,
        frame_record,
    ):
        """
        搜索下一个踢球者
        """
        # 5. 在ttl帧数窗口内探测下一个kicker
        probe_kicker = None
        if self.probe_kicker_up_frame_num > self.cur_frame_num:
            # 还没有超过probe_kicker的位置直接从bbox列表中获取即可
            for bbox in frame_record["bbox"]:
                if (self.probe_kicker_oid == bbox.oid and self.probe_kicker_cls == bbox.cls):
                    probe_kicker = bbox
                    break
        else:
            # 往前探测
            frame_probe_num = self.cur_frame_num + 1
            probe_ttl = self.PROBE_TTL
            while probe_ttl > 0 and (frame_probe_num in self.labels_dict.keys()):
                tmp_probe_kicker = self.labels_dict[frame_probe_num]["kicker"]
                if tmp_probe_kicker is not None:
                    if cur_kicker is None or (tmp_probe_kicker.oid != cur_kicker.oid or tmp_probe_kicker.cls != cur_kicker.cls):
                        self.probe_kicker_up_frame_num = frame_probe_num
                        self.probe_kicker_cls = tmp_probe_kicker.cls
                        self.probe_kicker_oid = tmp_probe_kicker.oid
                        break

                frame_probe_num += 1
                probe_ttl -= 1
        
        self.log(Video.DEBUG, "Frame " + str(self.cur_frame_num) + " probe next kicker finished.")
        return probe_kicker

    def __get_surroundings(
        self,
        cur_kicker,
        frame_record
    ):
        """
        根据当前踢球者绘制
        """
        surroundings = interaction.find_surroundings(cur_kicker, frame_record["bbox"], surrounding_max_dist_thres=self.SURROUNDING_MAX_DIST_THRES)
        self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " generate surroundings shape finished.")
        return surroundings
    
    def __render_team_shape(
        self,
        frame,
        surroundings,
    ):
        """
        绘制阵型
        """
        
        self_team_shape = team_shape.convexhull_calc(surroundings[0])
        enemy_team_shape = team_shape.convexhull_calc(surroundings[1])
        
        frame = render.renderTeamShape(frame, self_team_shape,(146,224,186))
        frame = render.renderTeamShape(frame, enemy_team_shape,(224,186,146))
        frame = render.renderRRectLabel_batch(frame, self_team_shape, (242, 168, 123))
        frame = render.renderRRectLabel_batch(frame, enemy_team_shape, (48, 96, 166))

        return frame, self_team_shape, enemy_team_shape

    def __render_distance_curve(
        self,
        frame,
        cur_kicker,
        team_shape,
        color = (16,255,16)
    ):
        """
        绘制距离曲线
        """
        # 4. 绘制当前kicker到其它队友或者是地方的一个距离 绘制曲线
        frame = render.renderDistance_batch(frame, cur_kicker, team_shape, color=color)
        self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " render team distance curve finished.")
        return frame
    
    def __render_kicker(
        self,
        frame,
        kicker,
        color = (255, 255, 255),
        font_color = (0, 0, 0)
    ):
        """
        绘制当前kicker
        """
        # 2. 如果当前帧存在kicker 则将当前帧的kicker给绘制出来
        frame = render.renderRRectLabel_batch(frame, [kicker], color=color, font_color=font_color, label_width=96, label_height=30)
        self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " render kicker finished.")
        return frame

    def __render_velocity(
        self,
        frame,
        kicker,
        velocity,
        color = (12,34,180)
    ):
        """
        绘制速度矢量
        """
        frame = render.renderArrow(frame, kicker, velocity, color)
        self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " render velocity finished.")
        return frame

    def __get_velocity(
        self,
        kicker,
    ):
        """
        获取速度矢量
        """
        velocity = None
        if kicker is None: return None

        if self.cur_frame_num + 2 < self.end_frame_num:
            dst_frame_record = self.labels_dict[self.cur_frame_num + 2]
            for bbox in dst_frame_record["bbox"]:
                if kicker.oid == bbox.oid and kicker.cls == bbox.cls:
                    # 仅仅考虑当前kicker的速度矢量
                    velocity = (bbox.xcenter - kicker.xcenter, bbox.ycenter - kicker.ycenter)
        self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " get kicker velocity finished.")
        return velocity

    def render_tactic(
        self,
        frame,
    ):
        """
        渲染得到的战术数据
        """
        if self.cur_frame_num not in self.tactics_map:
            return frame
        tactic = self.tactics_map[self.cur_frame_num]
        
        if tactic.tactic_type == "2-1":
            frame = self.render_21(frame, tactic)
        else:
            frame = self.render_32(frame, tactic)
        return frame

    def render_21(
        self,
        frame,
        tactic,
    ):
        """
        渲染2-1战术
        """
        frame_record = self.__load_frame_data()
        kicker1 = None
        kicker2 = None
        front_enmey = None
        for bbox in frame_record["bbox"]:
            if bbox.cls not in ["A", "B"]:
                continue
            if bbox.oid == tactic.kicker1_oid:
                kicker1 = bbox
            if bbox.oid ==  tactic.kicker2_oid:
                kicker2 = bbox
            if bbox.oid == tactic.front_oid:
                front_enmey = bbox
        
        # 渲染线条
        frame = render.renderTacticWithArrow_batch(frame, [kicker1, kicker2], color = (180,66,48))
        frame = render.renderTacticWithArrow_batch(frame, [front_enmey], color = (20,20,160))
        frame = render.renderTacticWithArrow_batch(frame, [kicker1, front_enmey], color = (0,160,160))

        return frame

    def render_32(
        self,
        frame,
        tactic,
    ):
        """
        绘制3-2战术
        """
        ... # TODO
        return frame

    def __render_tactic(
        self,
        frame,
        cur_kicker,
        velocity,
        surroundings,
    ):
        """
        绘制战术
        TODO 直接利用战术分析部分的数据而不是再分析一遍
        """
        self_player_bbox = []
        enemy_player_bbox = []
        front_player = None
        front_measure_score = 0

        # print("Surroundings:", len(surroundings[0]), len(surroundings[1]), cur_kicker, velocity)
        # 根据当前速度选择
        if velocity is not None:
            # 从同队中选择球员
            for bbox in surroundings[0]:
                # 计算是否与运动方向同向
                cosx = self.calc_cosx(bbox, cur_kicker, velocity)
                if (cosx is not None) and cosx > - 0.6:
                    # 计算像素距离
                    pixel_dist = interaction.calc_distance_in_pixel((cur_kicker.xcenter, cur_kicker.ycenter), (bbox.xcenter, bbox.ycenter))
                    self_player_bbox.append((bbox, cosx, pixel_dist))
            self.log(Video.DEBUG, "Frame " + str(self.cur_frame_num) + " get self-tactic finished.")

            # 从另外一队中先选择一个和当前kicker前方的球员
            for bbox in surroundings[1]:
                # 计算是否与运动方向同向
                cosx = self.calc_cosx(bbox, cur_kicker, velocity)
                if cosx is not None:
                    pixel_dist = interaction.calc_distance_in_pixel((cur_kicker.xcenter, cur_kicker.ycenter), (bbox.xcenter, bbox.ycenter))
                    tmp_score = cosx * (1 / math.exp(0.1 * pixel_dist))
                    if tmp_score > front_measure_score:
                        front_measure_score = tmp_score
                        front_player = bbox

            self.log(Video.DEBUG, "Frame " + str(self.cur_frame_num) + " get front player finished.")

            # 从另外一队中选择能够和front_player配合的球员
            if front_player is not None:
                for bbox in surroundings[1]:
                # 计算是否与运动方向同向
                    cosx = self.calc_cosx(bbox, front_player, velocity)
                    if cosx is not None and abs(cosx) <= 0.3:
                        pixel_dist = interaction.calc_distance_in_pixel((front_player.xcenter, front_player.ycenter), (bbox.xcenter, bbox.ycenter))
                        enemy_player_bbox.append((bbox, abs(cosx), pixel_dist))
            self.log(Video.DEBUG, "Frame " + str(self.cur_frame_num) + " get enemy-tactic finished.")

        # print(self_player_bbox)
        self_player_bbox = sorted(self_player_bbox, key=lambda x: x[1] * (1 / math.exp(0.1 * x[2])), reverse=True)
        self_render_bbox = [bbox for (bbox, _, _) in self_player_bbox]
        self_render_bbox.insert(0, cur_kicker)
        enemy_player_bbox = sorted(enemy_player_bbox, key=lambda x: -(x[1] * x[2]), reverse=True)
        enemy_render_bbox = [bbox for (bbox, _, _) in enemy_player_bbox]

        if front_player is not None: enemy_render_bbox.insert(0, front_player)
        
        if len(enemy_render_bbox) >= 2 and len(self_render_bbox) >= 3:
            # 3-2战术
            self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " 3-2 tactic finished.")
            self_render_bbox = self_render_bbox[:3]
            enemy_render_bbox = enemy_render_bbox[:2]
            self_render_bbox.append(cur_kicker)
        elif len(enemy_render_bbox) >= 1 and len(self_render_bbox) >= 2:
            # 2-1战术
            self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " 2-1 tactic finished.")
            self_render_bbox = self_render_bbox[:2]
            enemy_render_bbox = enemy_render_bbox[:1]
        else:
            # TODO 实现其余战术
            front_player = None
            ...

        # 战术绘制
        if front_player is not None:
            # frame = render.renderTactic_batch(frame, self_render_bbox, color = (180,66,48))
            # frame = render.renderTactic_batch(frame, enemy_render_bbox, color = (20,20,160))
            # frame = render.renderTactic_batch(frame, [cur_kicker, front_player], color = (0,160,160))
            frame = render.renderTacticWithArrow_batch(frame, self_render_bbox, color = (180,66,48))
            frame = render.renderTacticWithArrow_batch(frame, enemy_render_bbox, color = (20,20,160))
            frame = render.renderTacticWithArrow_batch(frame, [cur_kicker, front_player], color = (0,160,160))
            self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " render tactic finished.")

        return frame

    def get_one_rendered_frame(
        self, 
        btn_cfg,
    ):
        """
        绘制一帧画面的核心函数，主要用来完成一帧画面绘制时的各个流程
        """
        # 超过绘制边界
        if self.cur_frame_num not in self.labels_dict.keys() or self.cur_frame_num >= self.end_frame_num:
            return None

        frame = self.__load_frame()
        frame_record = self.__load_frame_data()
        ball = frame_record["ball"]
        cur_kicker = frame_record["kicker"]

        # 6 显示bbox
        if btn_cfg.show_bbox_flag:
            frame = self.__render_object_bbox(frame, frame_record)

        if ball is not None:
            
            # 1. 将识别到的足球给绘制出来. 标明位置
            if btn_cfg.show_ball_flag:
                frame = self.__render_ball(frame, ball)
            
            # 2. 探测下一个kicker
            probe_kicker = self.__find_next_kicker(cur_kicker, frame_record)

            if cur_kicker is not None:
                # 3. 将当前帧kicker的周围按照范围将所有的对象检测出来 绘制战术进攻阵型或者防守阵型 显然这里速度很慢 需要批处理 可以看作是一个凸包
                surroundings = self.__get_surroundings(cur_kicker, frame_record)
                if btn_cfg.show_shape_flag:
                    frame, self_team_shape, enemy_team_shape = self.__render_team_shape(frame, surroundings)
                    # 4. 绘制距离曲线
                    if btn_cfg.show_curve_flag:
                        frame = self.__render_distance_curve(frame, cur_kicker, self_team_shape, color=(16,240,64))
                        frame = self.__render_distance_curve(frame, cur_kicker, enemy_team_shape, color=(16,64,128))
                
                # 5. 绘制可能的kicker
                if btn_cfg.show_kicker_flag:
                    frame = self.__render_kicker(frame, cur_kicker, color=(255,255,255), font_color=(0, 0, 0))
                    if probe_kicker is not None:
                        frame = self.__render_kicker(frame, probe_kicker, color=(0,0,255), font_color=(0, 0, 0))

                # 7. 绘制kicker和下一个kicker运动速度示意以2帧后的位置为目标位置
                cur_velocity = self.__get_velocity(cur_kicker)
                probe_velocity = self.__get_velocity(probe_kicker)

                if btn_cfg.show_vel_flag:
                    if cur_velocity is not None:
                        frame = self.__render_velocity(frame, cur_kicker, cur_velocity, color = (240,129,12))
                    if probe_velocity is not None and probe_kicker is not None:
                        frame = self.__render_velocity(frame, probe_kicker, probe_velocity, color = (12,129,240))

                # 8. 绘制3-2战术或者是绘制2-1战术
                if btn_cfg.show_tactic_flag:
                    # 之前旧的可以转换为
                    # frame = self.__render_tactic(frame, cur_kicker, cur_velocity, surroundings)
                    frame = self.render_tactic(frame)
                
        self.cur_frame_num += 1
        if not btn_cfg.play_flag:
            self.cur_frame_num -= 1

        self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " finished.")
        return frame

    def extract_tactics_segments(self):
        """
        根据标签数据在初始化的过程中找出所有的战术片段 帧号区间
        这个函数可以看作是渲染一帧的镜像 耦合程度有点高 有点**代码的坏味道 有时间可以完全重构
        """
        tactics_records = {
            constant.TACTIC_21:[],   # 简单的二过一战术
            constant.TACTIC_32:[],   # 简单的三过二战术
        }

        frame_start_index = None
        tactic_type = None

        self.get_frames()

        for frame_id in self.labels_dict.keys():

            self.cur_frame_num = frame_id
            frame_record = self.__load_frame_data()

            cur_kicker = frame_record["kicker"]
            ball = frame_record["ball"]

            # 在ball和当前踢球者全存在的情况下进行挖掘
            if ball is not None and cur_kicker is not None:

                surroundings = self.__get_surroundings(cur_kicker, frame_record)
                cur_velocity = self.__get_velocity(cur_kicker)

                self_player_bbox = []
                enemy_player_bbox = []
                front_player = None
                front_measure_score = 0

                # 根据当前速度选择
                if cur_velocity is not None:
                    # 从同队中选择球员
                    for bbox in surroundings[0]:
                        # 计算是否与运动方向同向
                        cosx = self.calc_cosx(bbox, cur_kicker, cur_velocity)
                        if (cosx is not None) and cosx > - 0.6:
                            # 计算像素距离
                            pixel_dist = interaction.calc_distance_in_pixel((cur_kicker.xcenter, cur_kicker.ycenter), (bbox.xcenter, bbox.ycenter))
                            self_player_bbox.append((bbox, cosx, pixel_dist))

                    # 从另外一队中先选择一个和当前kicker前方的球员
                    for bbox in surroundings[1]:
                        # 计算是否与运动方向同向
                        cosx = self.calc_cosx(bbox, cur_kicker, cur_velocity)
                        if cosx is not None:
                            pixel_dist = interaction.calc_distance_in_pixel((cur_kicker.xcenter, cur_kicker.ycenter), (bbox.xcenter, bbox.ycenter))
                            tmp_score = cosx * (1 / math.exp(0.1 * pixel_dist))
                            if tmp_score > front_measure_score:
                                front_measure_score = tmp_score
                                front_player = bbox

                    # 从另外一队中选择能够和front_player配合的球员
                    if front_player is not None:
                        for bbox in surroundings[1]:
                        # 计算是否与运动方向同向
                            cosx = self.calc_cosx(bbox, front_player, cur_velocity)
                            if cosx is not None and abs(cosx) <= 0.3:
                                pixel_dist = interaction.calc_distance_in_pixel((front_player.xcenter, front_player.ycenter), (bbox.xcenter, bbox.ycenter))
                                enemy_player_bbox.append((bbox, abs(cosx), pixel_dist))
                
                # print(self_player_bbox)
                self_player_bbox = sorted(self_player_bbox, key=lambda x: x[1] * (1 / math.exp(0.1 * x[2])), reverse=True)
                self_render_bbox = [bbox for (bbox, _, _) in self_player_bbox]
                self_render_bbox.insert(0, cur_kicker)
                enemy_player_bbox = sorted(enemy_player_bbox, key=lambda x: -(x[1] * x[2]), reverse=True)
                enemy_render_bbox = [bbox for (bbox, _, _) in enemy_player_bbox]

                if front_player is not None: enemy_render_bbox.insert(0, front_player)

                if len(enemy_render_bbox) >= 2 and len(self_render_bbox) >= 3:
                    if tactic_type is None:  # 一段时间后首次检测出3-2
                        tactic_type = constant.TACTIC_32
                        frame_start_index = frame_id
                elif len(enemy_render_bbox) >= 1 and len(self_render_bbox) >= 2:
                    if tactic_type is None:  # 一段时间后首次检测出2-1
                        tactic_type = constant.TACTIC_21
                        frame_start_index = frame_id
                elif tactic_type is not None:
                    # 出现中断
                    tactics_records[tactic_type].append((frame_start_index, frame_id))
                    tactic_type = None
                    frame_start_index = None
                else:
                    # 没有检测出战术
                    tactic_type = None
                    frame_start_index = None
        
        # 过滤过短的不合法的长度
        short_thresh = 6
        longer_21_tactics = filter(lambda x: x[1] - x[0] >= short_thresh, tactics_records[constant.TACTIC_21])
        longer_32_tactics = filter(lambda x: x[1] - x[0] >= short_thresh, tactics_records[constant.TACTIC_32])

        def longer(seg):
            return (max(seg[0] - 5, 1), min(seg[1] + 5, self.get_frames()))

        longer_21_tactics = map(longer, longer_21_tactics)
        longer_32_tactics = map(longer, longer_32_tactics)

        video_tactics_segs = [[],[]]

        for tactic in longer_21_tactics:
            seg = self.copy_self()
            seg.tactic_type = constant.TACTIC_21
            seg.cur_frame_num = tactic[0]
            seg.start_frame_num = tactic[0]
            seg.end_frame_num = tactic[1]
            video_tactics_segs[0].append(seg)

        for tactic in longer_32_tactics:
            seg = self.copy_self()
            seg.tactic_type = constant.TACTIC_32
            seg.cur_frame_num = tactic[0]
            seg.start_frame_num = tactic[0]
            seg.end_frame_num = tactic[1]
            video_tactics_segs[1].append(seg)
        
        self.cur_frame_num = 1
        
        # 添加
        datahub.DataHub.add_tactics(self.name, video_tactics_segs)

    def extract_tactics_by_fsm(self):
        """
        利用FSM抽取2-1或者3-2战术
        """
        # TODO copy
        self.tactic_fsm.config(self.labels_dict)
        tactics_list = self.tactic_fsm.run()
        # fast cache
        self.tactics_map = {}
        for tactic in tactics_list:
            for i in range(tactic.start_frame_num, tactic.end_frame_num + 1):
                self.tactics_map[i] = tactic
        
    def copy_self(self):
        """
        产生一份自己的拷贝 以供修改
        """
        new_video = Video(self.name, status=1, is_seg=True)

        # 只读数据引用拷贝
        new_video.video_status = self.video_status
        new_video.upload_video_path = self.upload_video_path
        new_video.illegal_tactics_frames = self.illegal_tactics_frames
        new_video.labels_dict = self.labels_dict

        return new_video
        
    def calc_cosx(self, bboxa, bboxb, velocity):
        """
        计算连线的方向向量和速度之间的余弦值
        """
        line_vector = (bboxa.xcenter - bboxb.xcenter, bboxa.ycenter - bboxb.ycenter)
        if (line_vector[0] == 0 and line_vector[1] == 0) or (velocity[0] == 0 and velocity[1] == 0):
            return None
        # 计算是否与运动方向垂直
        inner_value = line_vector[0] * velocity[0] + line_vector[1] * velocity[1]
        norm = math.sqrt(line_vector[0] ** 2 + line_vector[1] ** 2) * math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        cosx = inner_value / norm
        return cosx

    def destroy(self):

        # 通知线程退出 清理资源
        # `TODO ??? 理论上使用下面的循环代码才是正确的优雅通知线程关闭 但是这里会处于死循环 初步猜测就是因为Python的GIL导致 Player
        # 中的draw线程也是如此`

        # while self.process_thread.is_alive():
        #     self.process_thread.stop()
        #     time.sleep(0.1)

        if self.process_thread is not None and self.process_thread.is_alive() and not self.process_thread.is_stop():
            self.process_thread.stop()