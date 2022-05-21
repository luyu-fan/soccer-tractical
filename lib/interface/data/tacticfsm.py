"""
简单战术识别有限状态机
"""

class TacticFSM:

    """
    用于战术分析挖掘的有限状态机
    """

    IDLE = 0               # 未激活的状态
    TACTIC_21_INIT = 1     # 初始化激活状态 即新传入了一个踢球者
    TACTIC_21_PRE = 2      # 前半阶段 即已经找到了两个同队的球员他们之间在踢球

    SEEK_FIRST_STAGET_KICKER = 0    # 寻找踢球者的第一个阶段 找到一个有效的踢球者即可
    SEEK_SECOND_STAGET_KICKER = 1    # 寻找踢球者的第二个阶段 找到和第一个踢球者同队的配合球员 完成战术的上部分目标搜寻
    SEEK_THIRD_STAGET_KICKER = 2  # 寻找踢球者的第三个阶段 找到第一个球员即可
    EXIT = 3
    SAVE = 4

    def __init__(self, labels_dict) -> None:

        # 跟踪结束后所有的标签数据
        self.labels_dict = labels_dict
        self.next_frame_id = 1

        self.fistr_kicker = None
        self.first_kicker_frame_id = None
        self.second_kicker = None
        self.second_kicker_frame_id = None

        self.seek_window_size = 30

        self.tactic_list = []    # 记录找到的所有战术对象


    def load_frame_data(self):
        """
        获取每一帧对应的数据
        """
        if self.next_frame_id in self.labels_dict:
            return self.labels_dict[self.next_frame_id]
        return None

    def run(self):
        """
        利用到的主要参数:

        detections: 视频片段中跟踪数据中的所有检测框

        挖掘主流程:

        1. 遍历循环 找到第一个踢球者

        2. 从这个踢球者出发 往下找新的踢球者

            1. 如果超过了挖掘窗口，退出
            2. 如果是这个踢球者自身，继续
            3. 如果是对方队伍，则更新踢球者重新在窗口内搜索
            4. 如果是同队队伍，则找到了第二个踢球者

        3. 从第二个踢球者出发 往下找新的踢球者

            1. 如果超过了挖掘窗口，退出
            2. 如果是这个踢球者自身，继续
            3. 如果是对方队伍，则回退到之前的状态更新踢球者重新在窗口内搜索
            4. 如果是同队队伍
                1. 是之前的那个踢球者，则是一段潜在的二过一战术
                2. 是新的踢球者，则有可能是一段潜在的三过二战术
            
        4. 如果是潜在的二过一战术
            1. 基本上不怎么判断的情况下就可以直接将其作为二过一结果
            2. 判断二过一的对方球员是谁
            3. 判断是否的确过了这个球员
        """

    def seek_kicker(self): 
        frame_record = self.load_frame_data()
        kicker = frame_record["kicker"] if frame_record is not None else None
        return kicker

    def seek_first_kicker(self):
        """
        搜索找到首个踢球者
        """
        kicker = None
        while kicker is None and self.next_frame_id in self.labels_dict.keys():
            kicker = self.seek_kicker()
            self.next_frame_id += 1
        if kicker is not None:
            self.first_kicker = kicker
            self.first_kicker_frame_id = self.next_frame_id - 1
            return TacticFSM.SEEK_SECOND_STAGET_KICKER
        else:
            self.first_kicker = None
            self.first_kicker_frame_id = None
            return TacticFSM.EXIT
        
    def seek_second_kicker(self):
        """
        找到第一个踢球者之后去搜索第二个踢球者
        """
        kicker = None
        seek_offset = 0
        while kicker is None and self.next_frame_id in self.labels_dict.keys() and seek_offset < self.seek_window_size:
            kicker = self.seek_kicker()
            self.next_frame_id += 1
            seek_offset += 1

        if kicker is None:  
            # 窗口内没有找到新的踢球者只能从第一个踢球者重新搜索
            self.first_kicker = None
            self.first_kicker_frame_id = None
            return TacticFSM.SEEK_FIRST_STAGET_KICKER
        elif kicker.oid == self.first_kicker.oid:
            # 说明第一个kicker踢球的时间比较长一直在占有球可以适当往后挪一下位置继续搜索即可
            if self.next_frame_id - self.first_kicker_frame_id >= self.seek_window_size // 2:
                self.first_kicker = kicker
                self.first_kicker_frame_id = self.next_frame_id - 1
            return TacticFSM.SEEK_SECOND_STAGET_KICKER
        else:
            # 如果找到了不相同的kicker则需要判断他们是否同属于同一个阵营
            if kicker.cls == self.first_kicker.cls:
                # 同一个阵营内则直接将其作为第二个kicker
                # TODO 对于那些不一定非要有回传的情况是否能够判断出来
                self.second_kicker = kicker
                self.second_kicker_frame_id = self.next_frame_id - 1
                return TacticFSM.SEEK_THIRD_STAGET_KICKER
            else:
                # 不是同一个阵营内则将其更新作为第一个kicker
                self.first_kicker = kicker
                self.first_kicker_frame_id = self.next_frame_id - 1
                return TacticFSM.SEEK_SECOND_STAGET_KICKER

    def seek_third_kicker(self):
        """
        搜索第三个kicker
        """
        kicker = None
        seek_offset = 0
        while kicker is None and self.next_frame_id in self.labels_dict.keys() and seek_offset < self.seek_window_size:
            kicker = self.seek_kicker()
            self.next_frame_id += 1
            seek_offset += 1

        if kicker is None:  
            # 窗口内没有找到新的踢球者只能从第一个踢球者重新搜索
            self.first_kicker = None
            self.first_kicker_frame_id = None
            return TacticFSM.SEEK_FIRST_STAGET_KICKER
        elif kicker.oid == self.first_kicker.oid:
            # 如果第一个kicker重新出现说明可能已经完成了一个战术动作
            # TODO 判断是否有越过某个对方球员才能作为新的条件
            tactic_21_result = (self.first_kicker_frame_id, self.second_kicker_frame_id, self.next_frame_id - 1)
            self.tactic_list.append(tactic_21_result)
            return TacticFSM.SEEK_FIRST_STAGET_KICKER
        elif kicker.oid == self.second_kicker.oid:
            # 继续搜索
            return TacticFSM.SEEK_THIRD_STAGET_KICKER
        else:
            # 如果找到了不相同的kicker则需要判断他们是否同属于同一个阵营
            if kicker.cls == self.first_kicker.cls:
                # 同一个阵营内则更新前两个kicker使其继续搜索
                self.first_kicker = self.second_kicker
                self.first_kicker_frame_id = self.second_kicker_frame_id
                self.second_kicker = kicker
                self.second_kicker_frame_id = self.next_frame_id - 1
                return TacticFSM.SEEK_THIRD_STAGET_KICKER
            else:
                # 不是同一个阵营内则将其更新作为第一个kicker
                self.first_kicker = kicker
                self.first_kicker_frame_id = self.next_frame_id - 1
                return TacticFSM.SEEK_SECOND_STAGET_KICKER
    
    def fuse_32(self):
        """
        搜索是否存在两个二过一融合位一个三过二
        """
        

