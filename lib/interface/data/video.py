"""
对视频片段数据及处理的封装
"""
import os
import math
import threading
from typing import Iterable
import cv2

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

from lib.interface.common import datahub
 
from lib.net import client

class FSM:
    """
    用于数据自动处理的有限状态机
    """
    START = 0
    SEEK = 1
    CANDIDATE = 2
    FIX = 3
    FINISH = 4
    
    def __init__(
        self,
        window_size = 10,
        activate_thres = 0.6,
        frames_result = None,
    ) -> None:
        """
        Args:
            window_size: 状态机搜素的时间窗口
            activate_thres: 由候选状态转变为确定状态的阈值
            frames_result: 所有的帧数据
        """

        assert isinstance(frames_result, Iterable)
        
        # 状态字典
        self.window = Window(frames_result, window_size, activate_thres)
        self.states_map = {
            FSM.SEEK: SeekState(),
            FSM.CANDIDATE: CandidateState(),
            FSM.FIX: FixState(),
        }

    def run(self):
        """
        FSM后修复检测的主循环
        """
        next_state_name = FSM.SEEK
        while True:
            cur_state = self.states_map[next_state_name]
            next_state_name = cur_state.do(self.window)
            if next_state_name == FSM.FINISH:
                return
            
class Window:

    def __init__(
        self,
        frames_result,
        size,
        activate_thres,
    ) -> None:

        assert isinstance(frames_result, Iterable)

        # 所有帧的结果数组
        self.frames_result = frames_result
        self.total_num = len(frames_result)
        # 窗口大小
        self.size = size
        # 候选激活比例阈值
        self.activate_thres = activate_thres
        # 相对于全部检测帧的起始位置
        self.start_index = 0
        # 最新的已经处于修复之后的索引
        self.fixed_index = 0
        # 修复所用到的关键检测索引
        self.anchor_index = None

    def get_total_result_num(self):
        return self.total_num

    def check_linear_neighbor(
        self,
        i,
        dxdy,
        j,         
    ):
        """
        检查是否符合线性位置
        Args:
            i: 窗口内第i个检测
            dxdy: 以第i个检测为anchor时的变化量
            j: 第j个检测
        """
        if i == j: return True
        elif i < j:
            x_dist = self.frames_result[j][2] - self.frames_result[i][2]
            y_dist = self.frames_result[j][3] - self.frames_result[i][3]
        else:
            x_dist = self.frames_result[i][2] - self.frames_result[j][2]
            y_dist = self.frames_result[i][3] - self.frames_result[j][3]
        if x_dist <= (abs(j - i) * dxdy[0] + self.frames_result[i][4]) and y_dist <= (abs(j - i) * dxdy[1] + self.frames_result[i][5]):
            return True
        else:
            return False
            
    def get_det(self, index):
        """
        获取指定位置的检测结果
        """
        if index < self.total_num:
            return self.frames_result[index]
        else:
            raise IndexError

    def get_start_index(self):
        """
        返回当前的起始索引
        """
        return self.start_index

    def move(
        self,
        step = 1,
    ):
        """
        移动窗口
        """
        self.start_index += step 
        # self.inner_array = self.frames_result[self.start_index:self.start_index + self.size]

    def calc_dxdy(self, det_t_1, det_t):
        """
        计算相邻两个检测的中心点的dxdy 实际上为水平和竖直方向上的移动速度
        Args:
            det_t_1: t-1时刻的检测 x1y1wh
            det_t: t时刻的检测
        TODO 此处仅仅考虑两个位置 后续可以考虑更准确的形式
        """
        dx = det_t[2] - det_t_1[2]
        dy = det_t[3] - det_t_1[3]
        return [dx, dy]

    def get_activate_thres(self):
        return self.activate_thres

    def config_fix(
        self,
        anchor_index,
        dxdy,
    ):
        """
        配置修复时所需要的一些必要数据
        Args:
            anchor_index: 最佳的用于修正轨迹采用的锚检测位置
            dxdy: anchor对应的变化速度
        """
        self.anchor_index = anchor_index
        self.anchor_dxdy = dxdy

    def discard_det(self, index):
        """
        抛弃index位置的检测，前提是这个检测并不处于前一个检测窗口中
        """
        self.frames_result[index] = None

    def fix(self, index):
        """
        尝试性修复窗口内的某个检测结果
        """
        if self.frames_result[index] is None or \
            not self.check_linear_neighbor(self.anchor_index, self.anchor_dxdy, index):
            if index > self.anchor_index:
                x1 = self.frames_result[self.anchor_index][2] + self.anchor_dxdy[0] * (index - self.anchor_index)
                y1 = self.frames_result[self.anchor_index][3] + self.anchor_dxdy[1] * (index - self.anchor_index)
                w = self.frames_result[self.anchor_index][4]
                h = self.frames_result[self.anchor_index][5]
                # 产生一个修复后的检测框
                pre = self.frames_result[index]
                self.frames_result[index] = ["Ball", 1, x1, y1, w, h]
                # print("Fix:", pre, self.frames_result[index])
            else:
                x1 = self.frames_result[self.anchor_index][2] - self.anchor_dxdy[0] * (index - self.anchor_index)
                y1 = self.frames_result[self.anchor_index][3] - self.anchor_dxdy[1] * (index - self.anchor_index)
                w = self.frames_result[self.anchor_index][4]
                h = self.frames_result[self.anchor_index][5]
                # 产生一个修复后的检测框
                pre = self.frames_result[index]
                self.frames_result[index] = ["Ball", 1, x1, y1, w, h]
                # print("Fix:", pre, self.frames_result[index])
        else:
            return

class ProcessState:
    """
    后处理的状态父类
    """
    def __init__(
        self,
    ):
        ...

    def do(self, window):
        raise NotImplemented

class SeekState(ProcessState):
    """
    寻找一个合适的窗口 使其处于候状态
    最简单的状态 实际上就是找到一个以非空检测结果为开始的窗口。
    """
    def __init__(self):
        super(SeekState, self).__init__()

    def do(self, window):
        """
        寻找合适窗口并激活的处理逻辑
        """
        assert isinstance(window, Window)

        # 找到第一个非空检测即可
        probe_index = window.get_start_index()
        while probe_index < window.get_total_result_num():
            det =  window.get_det(probe_index)
            if det is not None:
                window.move(probe_index - window.get_start_index())
                return FSM.CANDIDATE
            else:
                probe_index  += 1
        
        return FSM.FINISH

class CandidateState(ProcessState):
    """
    判断当前窗口内的各个检测时候能够构成一段稳定的轨迹
    具体方法是通过循环计算各个检测于其它检测的距离
    计算过程为O(n2)复杂度
    1. 首先需要利用t和t-1时刻的位置计算两侧偏移的速度，两个方向。
    2. 根据相应的检测所对应的位置索引，计算对应检测是否满足线性关系，即在一定的范围内。
    3. 统计各个检测所构成的满足线性要求的比例，超过一定的比例则满足激活。
    4. 以最大比例的检测为锚，进入到修正环节。 
    """
    def __init__(self):
        super(CandidateState, self).__init__()

    def do(self, window):
        """
        寻找合适窗口并激活的处理逻辑
        """
        assert isinstance(window, Window)

        # 窗口中实际容量仅为1
        if window.get_start_index() >= window.get_total_result_num() - 1:
            # 忽略掉直接返回完成状态
            # TODO 应该考虑更详细的边界情况
            return FSM.FINISH

        max_rate = -1
        max_index = -1
        best_dxdy = None

        start = window.get_start_index()
        end = min(start + window.size, window.get_total_result_num())
        for i in range(start, end):
            if window.frames_result[i] is None: continue
            # 处理边界条件
            if i == start:
                # 窗口的左侧
                if window.frames_result[i + 1] is None:
                    continue
                else:
                    dxdy = window.calc_dxdy(window.frames_result[i], window.frames_result[i + 1])
            elif i == end - 1:
                # 窗口的右侧
                if window.frames_result[i - 1] is None:
                    continue
                else:
                    dxdy = window.calc_dxdy(window.frames_result[i - 1], window.frames_result[i])
            else:
                # 中间位置
                if window.frames_result[i - 1] is None and window.frames_result[i + 1] is None:
                    continue
                elif window.frames_result[i - 1] is not None:
                    dxdy = window.calc_dxdy(window.frames_result[i - 1], window.frames_result[i])
                else:
                    dxdy = window.calc_dxdy(window.frames_result[i], window.frames_result[i + 1])

            # 迭代计算是否符合线性模型
            checked_calculator = 0
            for j in range(start, end):
                if window.frames_result[j] is None: continue
                if window.check_linear_neighbor(i, dxdy, j):
                    checked_calculator += 1
            
            # 符合线性条件的比例
            rate = (checked_calculator / (end - start))
            if rate >= window.get_activate_thres() and rate > max_rate:
                max_rate = rate
                max_index = i
                best_dxdy = dxdy

        # 如果当前窗口满足轨迹的连续性则返回修复状态 否则丢弃窗口左侧的检测进行下一个窗口探测
        if best_dxdy is not None:
            window.config_fix(max_index, best_dxdy)
            return FSM.FIX
        else:
            window.discard_det(window.get_start_index())
            return FSM.SEEK

class FixState(ProcessState):
    """
    修复状态，以锚检测所在位置为准进行修正，即将窗口内为空或者为不满足线性条件的检测进行修正和替换。
    时间复杂度为O(n)
    修正的同时需要标记窗口内的每一个候选框。同时将seek指针移动到窗口的60%处，使其能够进行下一轮修复。
    """
    def __init__(self):
        super(FixState, self).__init__()

    def do(self, window):
        """
        在窗口内按照指定的anchor进行修复，由于此时需要进行修改数据，因此需要在原始的传入的数组上进行修改
        """
        assert isinstance(window, Window)

        start_index = window.get_start_index()
        end = min(start_index + window.size, window.total_num)
        for k in range(start_index, end):
            # 对第k个进行尝试性修复
            window.fix(k)
            if window.fixed_index < k:
                window.fixed_index = k
                
        # 移动窗口
        window.move(window.size // 2 + 1)
        return FSM.SEEK
        
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
        # TODO 作为可以调节的参数暴露在GUI上
        self.KICK_DIST_PIXEL_THRES = 60
        self.SURROUNDING_MAX_DIST_THRES = 300
        self.PROBE_TTL = 45

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
        
        fsm = FSM(
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


    def __load_frame_data(self):
        """
        加载绘制时的每一帧数据
        """
        frame_record = self.labels_dict[self.cur_frame_num]
        img_path = os.path.join(constant.DATA_ROOT, "images", self.name.split(".")[0], "{:06d}.jpg".format(self.cur_frame_num))
        frame = cv2.imread(img_path)
        ball = frame_record["ball"]
        cur_kicker = frame_record["kicker"]

        self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " data prepared.")

        return ball, cur_kicker, frame, frame_record

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
        if self.cur_frame_num + 2 < self.total_frames:
            dst_frame_record = self.labels_dict[self.cur_frame_num + 2]
            for bbox in dst_frame_record["bbox"]:
                if kicker.oid == bbox.oid and kicker.cls == bbox.cls:
                    # 仅仅考虑当前kicker的速度矢量
                    velocity = (bbox.xcenter - kicker.xcenter, bbox.ycenter - kicker.ycenter)
        self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " get kicker velocity finished.")
        return velocity

    def __render_tractical(
        self,
        frame,
        cur_kicker,
        velocity,
        surroundings,
    ):
        """
        绘制战术
        """
        self_player_bbox = []
        enemy_player_bbox = []
        front_player = None
        front_cosx = 0
        # 根据当前速度选择
        if velocity is not None:
            # 从同队中选择球员
            for bbox in surroundings[0]:
                # 计算是否与运动方向同向
                cosx = self.calc_cosx(bbox, cur_kicker, velocity)
                if (cosx is not None) and (abs(cosx) > 0.6):
                    self_player_bbox.append((bbox, cosx))
            self.log(Video.DEBUG, "Frame " + str(self.cur_frame_num) + " get self-tractical finished.")

            # 从另外一队中先选择一个和当前kicker前方的球员
            for bbox in surroundings[1]:
                # 计算是否与运动方向同向
                cosx = self.calc_cosx(bbox, cur_kicker, velocity)
                if cosx is not None and cosx > front_cosx:
                    front_cosx = cosx
                    front_player = bbox

            self.log(Video.DEBUG, "Frame " + str(self.cur_frame_num) + " get front player finished.")
            # 从另外一队中选择能够和front_player配合的球员
            if front_player is not None:
                for bbox in surroundings[1]:
                # 计算是否与运动方向同向
                    cosx = self.calc_cosx(bbox, front_player, velocity)
                    if cosx is not None and abs(cosx) <= 0.3:
                        enemy_player_bbox.append((bbox, abs(cosx)))
            self.log(Video.DEBUG, "Frame " + str(self.cur_frame_num) + " get enemy-tractical finished.")

        # print(self_player_bbox)
        self_player_bbox = sorted(self_player_bbox, key=lambda x: x[1])
        self_render_bbox = [bbox for (bbox, _) in self_player_bbox]
        self_render_bbox.insert(0, cur_kicker)
        enemy_player_bbox = sorted(enemy_player_bbox, key=lambda x: -x[1])
        enemy_render_bbox = [bbox for (bbox, _) in enemy_player_bbox]
        if front_player is not None: enemy_render_bbox.insert(0, front_player)

        if len(enemy_render_bbox) >= 2 and len(self_render_bbox) >= 3:
            # 3-2战术
            self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " 3-2 tractical finished.")
            self_render_bbox = self_render_bbox[:3]
            enemy_render_bbox = enemy_render_bbox[:2]
            self_render_bbox.append(cur_kicker)
        elif len(enemy_render_bbox) >= 1 and len(self_render_bbox) >= 2:
            # 2-1战术
            self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " 2-1 tractical finished.")
            self_render_bbox = self_render_bbox[:2]
            enemy_render_bbox = enemy_render_bbox[:1]
        else:
            # TODO 实现其余战术
            front_player = None
            ...

        # 战术绘制
        if front_player is not None:
            # frame = render.renderTractical_batch(frame, self_render_bbox, color = (180,66,48))
            # frame = render.renderTractical_batch(frame, enemy_render_bbox, color = (20,20,160))
            # frame = render.renderTractical_batch(frame, [cur_kicker, front_player], color = (0,160,160))
            frame = render.renderTracticalWithArrow_batch(frame, self_render_bbox, color = (180,66,48))
            frame = render.renderTracticalWithArrow_batch(frame, enemy_render_bbox, color = (20,20,160))
            frame = render.renderTracticalWithArrow_batch(frame, [cur_kicker, front_player], color = (0,160,160))
            self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " render tractical finished.")

        return frame

    def get_one_rendered_frame(
        self, 
        btn_cfg,
    ):
        """
        绘制一帧画面的核心函数，主要用来完成一帧画面绘制时的各个流程
        """
        if self.cur_frame_num not in self.labels_dict.keys():
            return None

        ball, cur_kicker, frame, frame_record = self.__load_frame_data()

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

                # 方法: 以当前的kicker为参考 找到与其运动方向垂直且处在同一个运动的同队球员
                # 如果这样的球员能找到三个 则先尝试在运动方向上能不能找到和其相向的最近的两个对方球员 此时则可以完成3-2战术
                # 如果对方只有一个 则尝试绘制2-1战术
                # 如果对方没有 则放弃绘制
                # 8. 绘制3-2战术或者是绘制2-1战术
                if btn_cfg.show_tactic_flag:
                    frame = self.__render_tractical(frame, cur_kicker, cur_velocity, surroundings)
                
        self.cur_frame_num += 1
        if not btn_cfg.play_flag:
            self.cur_frame_num -= 1

        self.log(Video.INFO, "Frame " + str(self.cur_frame_num) + " finished.")
        return frame

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

        if self.process_thread.is_alive() and not self.process_thread.is_stop():
            self.process_thread.stop()