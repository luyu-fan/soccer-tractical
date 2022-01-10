"""
对视频片段数据及处理的封装
"""
import os
import threading
import cv2
import time
import numpy as np

from PIL import Image
from sklearn.cluster import KMeans

from lib.constant import constant
from lib.dataprocess import check_exists
from lib.dataprocess import prepare
from lib.render import render
from lib.interaction import interaction
from lib.utils import team_shape
from lib.workthread import thread as wthread
from lib.coloring import rgb2cn

from lib.net import client

class Video:

    LOADED = 0                     # 可直接播放的分析视频已加载
    UNPROCESS = 1                  # 数据未处理
    EXTRACT = 2                    # 抽帧
    TRACKING = 3                   # 跟踪
    INTERMEDIATE = 4               # 中间数据处理
    FINISHED = 5                   # 数据已经过服务器处理

    def __init__(
        self,
        name,
        status = 1,
        upload_video_path = None,
    ):
        """
        Args:
            name: 名称
            status: 状态 可以在初始化时设定 0未完成 1完成 2中间数据处理
            upload_video_path: 对应的原始上传的视频文件路径
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

        # 和视频播放相关的控制
        self.KICK_DIST_PIXEL_THRES = 60
        self.SURROUNDING_MAX_DIST_THRES = 400
        self.PROBE_TTL = 60

        # 视频序列往前探测分析对象时所需要做的标记
        self.probe_kicker_cls = ""
        self.probe_kicker_oid = ""
        self.probe_kicker_up_frame_num = 0
        self.cur_frame_num = 1

        # 空间状态更新句柄
        self.cover_update_handler = None
        self.status_update_handler = None

        # 网络客户端 (将视频处理任务看作是互相隔离的客户端)
        self.client = client.MOTClient(constant.REMOTE_IP, constant.REMOTE_PORT) if self.video_status == Video.UNPROCESS else None

        # 数据处理线程
        self.process_thread = wthread.WorkThread("video_process:"+self.name, self.process)
        self.process_thread.start()

        # 处理结果列表
        self.results_list = []
    
    def process(self, stop_event):
        """
        在视频展示之间完成一些准备工作。
        已处理完成视频: 根据跟踪结果完成绘制所需要数据的生成
        新上传视频片段: 跟踪以及后续处理
        Args:
            stop_event: 退出事件信号
        """
        if self.get_status() == Video.LOADED:
            self.process_loaded(stop_event)
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
        self.extract_frames()
        if stop_event.is_set(): return
        self.main_track(stop_event)
        if stop_event.is_set(): return
        # 3. 中间数据处理
        # TODO 足球轨迹修正算法
        try:
            if self.status_update_handler is not None:  self.status_update_handler("状态: 中间数据处理")
        except Exception as e:
            ...
        self.set_status(Video.INTERMEDIATE)
        # 4. 自动颜色分队算法
        self.coloring(stop_event)
        if stop_event.is_set(): return
        # 5. 写入label文件
        self.save_labels()
        self.set_status(Video.FINISHED)
        try:
            if self.status_update_handler is not None:  self.status_update_handler("状态: 处理完毕")
        except Exception as e:
            ...
    
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
            ball_result = result["ball"]
            for bbox, oid in zip(player_result[0], player_result[1]):
                x1y1wh_box = ["", oid, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                frame_result.append(x1y1wh_box)
                # cv2.rectangle(frame, (x1y1wh_box[2], x1y1wh_box[3]), (x1y1wh_box[2] + x1y1wh_box[4], x1y1wh_box[3] + x1y1wh_box[5]), color=(23,45,67), thickness=2)
            for bbox, oid in zip(ball_result[0], ball_result[1]):
                x1y1wh_box = ["Ball", oid, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                frame_result.append(x1y1wh_box)
                # cv2.rectangle(frame, (x1y1wh_box[2], x1y1wh_box[3]), (x1y1wh_box[2] + x1y1wh_box[4], x1y1wh_box[3] + x1y1wh_box[5]), color=(12,45,240), thickness=2)
            self.results_list.append(frame_result)
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
            for (frame_id, frame_mot_result) in enumerate(self.results_list):
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

    def move_to_loaded(self):
        """
        移动至已处理队列
        """
    
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
        index = 0
        for frame_mot_result, image_name in zip(self.results_list, os.listdir(self.imgs_folder_name)):
            # 加载图像
            img_path = os.path.join(self.imgs_folder_name, image_name)
            frame = cv2.imread(img_path)
            for i in range(len(frame_mot_result)):
                if stop_event.is_set(): return
                if frame_mot_result[i][0] == "Ball":
                    continue
                else:
                    x1 = max(0, frame_mot_result[i][2])
                    y1 = max(0, frame_mot_result[i][3])
                    x2 = x1 + frame_mot_result[i][4]
                    y2 = y1 + frame_mot_result[i][5]
                    obj_cn_rep = rgb2cn.get_img_mean_rep(frame[y1:y2, x1:x2, :].copy(), remove_green=True)
                    obj_cn_reps.append(obj_cn_rep)
                    index += 1

        # TODO 选择全部的样本非常慢 实际上可以考虑采用部分目标进行K-Means聚类 剩余的部分仅仅根据聚类结果参与分配
        # 先利用前若干帧中的对象形成的CN特征进行K-Means聚类，然后后续所有的对象目标计算与这个颜色中心的距离以此作为自己的队伍颜色划分
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(obj_cn_reps)

        # print(kmeans.labels_)
        # print(kmeans.cluster_centers_)

        index = 0
        for frame_mot_result in self.results_list:
            for i in range(len(frame_mot_result)):
                if stop_event.is_set(): return
                if frame_mot_result[i][0] == "Ball":
                    continue
                elif kmeans.labels_[index] == 0:
                    # print("A")
                    frame_mot_result[i][0] = "A"   # 分配给A队
                else:
                    frame_mot_result[i][0] = "B"   # 分配给B队
                    # print("B")
                index += 1

        # print(self.results_list)
            
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
            imgs_names = os.listdir(self.imgs_folder_name)
            if len(imgs_names) > 0:
                img_path = os.path.join(self.imgs_folder_name, imgs_names[0])
                self.cover_img = Image.open(img_path)
        return self.cover_img

    def get_frames(self):
        """
        返回视频片段的总帧数
        """
        if check_exists(self.imgs_folder_name):
            self.total_frames = len(os.listdir(self.imgs_folder_name))
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
        self.cur_frame_num = max(1, self.cur_frame_num - n)
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
        self.cur_frame_num = min(self.total_frames, self.cur_frame_num + n)
        self.probe_kicker_cls = ""
        self.probe_kicker_cls = ""
        self.probe_kicker_oid = ""
        self.probe_kicker_up_frame_num = 0

    def get_one_rendered_frame(
        self, 
        do_not_incr = False,
        show_bbox = False
    ):
        """
        绘制一帧画面的核心函数，主要用来
        """
        if self.cur_frame_num not in self.labels_dict.keys():
            return None
        frame_record = self.labels_dict[self.cur_frame_num]
        img_path = os.path.join(constant.DATA_ROOT, "images", self.name.split(".")[0], "{:06d}.jpg".format(self.cur_frame_num))
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
            if self.probe_kicker_up_frame_num > self.cur_frame_num:
                for bbox in frame_record["bbox"]:
                    if (self.probe_kicker_oid == bbox.oid and self.probe_kicker_cls == bbox.cls):
                        frame = render.renderRRectLabel_batch(frame, [bbox], color=(0, 0, 255), font_color=(0, 0, 0), label_width=96, label_height=30)
                        break
            else:
                frame_probe_num = self.cur_frame_num + 1
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

        self.cur_frame_num += 1
        if do_not_incr:
            self.cur_frame_num -= 1

        return frame

    def destroy(self):

        # 通知线程退出 清理资源
        # `TODO ??? 理论上使用下面的循环代码才是正确的优雅通知线程关闭 但是这里会处于死循环 初步猜测就是因为Python的GIL导致 Player
        # 中的draw线程也是如此`

        # while self.process_thread.is_alive():
        #     self.process_thread.stop()
        #     time.sleep(0.1)

        if self.process_thread.is_alive() and not self.process_thread.is_stop():
            self.process_thread.stop()