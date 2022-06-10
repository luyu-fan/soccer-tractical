"""
简单战术识别有限状态机
"""
import math
from lib.constant import constant

from lib.interaction import interaction


class Tactic:
    """
    简单战术基类
    """
    def __init__(self, start_frame_num, end_frame_num, tactic_type):
        self.start_frame_num = start_frame_num
        self.end_frame_num = end_frame_num
        self.tactic_type = tactic_type

class Tactic21(Tactic):
    """
    简单二过一战术
    """
    def __init__(
        self, 
        start_frame_num, 
        end_frame_num, 
        tactic_type,
        kicker2_frame_num, 
        kicker1_id,
        kicker2_id,
        front_id,
    ):
        super().__init__(start_frame_num, end_frame_num, tactic_type)
        """
        简单的二过一战术所需要的一些数据
        """
        self.kicker2_frame_num = kicker2_frame_num
        self.tactic_type = tactic_type
        self.kicker1_oid = kicker1_id
        self.kicker2_oid = kicker2_id
        self.front_oid = front_id

class Tactic32(Tactic):
    """
    简单二过一战术
    """
    def __init__(
        self, 
        start_frame_num, 
        end_frame_num, 
        tactic_type,
        first_kicker,
        first_kicker_frame_id,
        second_kicker,
        second_kicker_frame_id,
        third_kicker,
        third_kicker_frame_id,
        first_front,
        first_front_frame_id,
        second_front,
        second_front_frame_id,
        remain_kickers,
    ):
        super().__init__(start_frame_num, end_frame_num, tactic_type)
        """
        简单的三过二战术所需要的一些数据
        """
        
        self.first_kicker = first_kicker
        self.first_kicker_frame_id = first_kicker_frame_id
        self.second_kicker = second_kicker
        self.second_kicker_frame_id = second_kicker_frame_id
        self.third_kicker = third_kicker
        self.third_kicker_frame_id = third_kicker_frame_id
        self.first_front = first_front
        self.first_front_frame_id = first_front_frame_id
        self.second_front = second_front
        self.second_front_frame_id = second_front_frame_id
        self.remain_kickers = remain_kickers

class TacticFSM:

    def __init__(self) -> None:
        self.next_frame_id = 1
        self.window_base_index = None
        self.window_cur_index = None

        self.seek_window_size = 45
        self.surroundings_max_thres = 600

    def config(self, labels_dict):
        # 跟踪结束后所有的标签数据
        self.labels_dict = labels_dict

    def load_frame_data(self, index = None):
        """
        获取每一帧对应的数据
        """
        if index is None:
            index = self.next_frame_id
        if index in self.labels_dict:
            return self.labels_dict[index]
        return None
    
    def get_front_player(self, first_kicker, kickers_direction, surroundings):
        """
        根据阵型计算此时的对方球员
        """
        front_player = None
        front_measure_score = 0
        for bbox in surroundings:
            # 计算kicker1和kicker2传递球的路线中最靠近的那个点 即这个时候为运动过程中最可能的拦截球员
            tmp_direction = self.get_direction(first_kicker, bbox)
            cosx = self.get_cosx(kickers_direction, tmp_direction)
            if cosx is not None:
                pixel_dist = interaction.calc_distance_in_pixel((first_kicker.xcenter, first_kicker.ycenter), (bbox.xcenter, bbox.ycenter))
                tmp_score = cosx * (1 / math.exp(0.1 * pixel_dist))
                if tmp_score > front_measure_score:
                    front_measure_score = tmp_score
                    front_player = bbox
        return front_player

    def judge_point_positions(self, first_kicker, second_kicker, third_kicker, front_player):
        """
        判断点是否在三角形内部
        """
        if front_player is None:
            return False

        kicker_12_direction = self.get_direction(first_kicker, second_kicker)
        kicker_21_direction = self.get_direction(second_kicker, third_kicker)
        kicker_11_direction = self.get_direction(third_kicker, first_kicker)

        kicker_1front_direction = self.get_direction(first_kicker, front_player)
        kicker_2front_direction = self.get_direction(second_kicker, front_player)
        kicker_3front_direction = self.get_direction(third_kicker, front_player)

        v1 = self.get_cross_product(kicker_12_direction, kicker_1front_direction)
        v2 = self.get_cross_product(kicker_21_direction, kicker_2front_direction)
        v3 = self.get_cross_product(kicker_11_direction, kicker_3front_direction)

        # print("k12", v1)
        # print("k21", v2)
        # print("k11", v3)

        # 因为计算外积时使用的是右手螺旋定理所以应该所有值小于0才能说明是在三角形包围的内部
        # 理论上是不能等于0的但是由于摄像机镜头的移动导致的误差 当大于0的时候反而能将一些战术给检测出来???
        return (v1 <= 0 and v2 <= 0 and v3 <= 0) or (v1 > 0 and v2 > 0 and v3 > 0)

    def get_distance(self, bboxa, bboxb):
        """
        计算目标间的像素距离
        """
        return interaction.calc_distance_in_pixel((bboxa.xcenter, bboxa.ycenter), (bboxb.xcenter, bboxb.ycenter))

    def get_cross_product(self, directiona, directionb):
        """
        计算两个方向向量的叉乘(二维)
        """
        # 代表了外积向量的模
        return directiona[0] * directionb[1] - directiona[1] * directionb[0]

    def get_direction(
        self,
        bboxa,
        bboxb,
    ):
        """
        获取两个球员之间的方向向量
        """
        # 仅仅考虑当前kicker的速度矢量
        direction = (bboxb.xcenter - bboxa.xcenter, bboxb.ycenter - bboxa.ycenter)
        return direction

    def get_cosx(self, directiona, directionb):
        """
        计算两个方向向量之间的方向向量对应的余弦距离
        """
        if (directiona[0] == 0 and directiona[1] == 0) or (directionb[0] == 0 and directionb[1] == 0):
            return None
        # 计算是否与运动方向垂直
        inner_value = directiona[0] * directionb[0] + directiona[1] * directionb[1]
        norm = math.sqrt(directiona[0] ** 2 + directiona[1] ** 2) * math.sqrt(directionb[0] ** 2 + directionb[1] ** 2)
        cosx = inner_value / norm
        return cosx


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

    def get_surroundings(
        self,
        cur_kicker,
        frame_record
    ):
        """
        根据当前踢球者绘制
        """
        surroundings = interaction.find_surroundings(cur_kicker, frame_record["bbox"], surrounding_max_dist_thres=self.surroundings_max_thres)
        return surroundings


class Tactic21FSM(TacticFSM):

    """
    用于战术分析挖掘的有限状态机(仅仅用于2-1战术挖掘)
    """
    SEEK_FIRST_STAGET_KICKER = 0       # 寻找踢球者的第一个阶段 找到一个有效的踢球者即可
    SEEK_SECOND_STAGET_KICKER = 1      # 寻找踢球者的第二个阶段 找到和第一个踢球者同队的配合球员 完成战术的上部分目标搜寻
    SEEK_THIRD_STAGET_KICKER = 2       # 寻找踢球者的第三个阶段 找到第一个球员即可
    EXIT = 3

    def __init__(self) -> None:

        super(Tactic21FSM, self).__init__()
        
        self.fistr_kicker = None
        self.first_kicker_frame_id = None
        self.second_kicker = None
        self.second_kicker_frame_id = None

        self.tactic_list = []    # 记录找到的所有战术对象
        self.fsm_state = Tactic21FSM.SEEK_FIRST_STAGET_KICKER
        self.fsm_state_map = {
            Tactic21FSM.SEEK_FIRST_STAGET_KICKER: self.seek_first_kicker,
            Tactic21FSM.SEEK_SECOND_STAGET_KICKER: self.seek_second_kicker,
            Tactic21FSM.SEEK_THIRD_STAGET_KICKER: self.seek_third_kicker,
        }

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
            
        4. 如果是潜在的二过一战术
            1. 基本上不怎么判断的情况下就可以直接将其作为二过一结果
            2. 判断二过一的对方球员是谁
            3. 判断是否的确过了这个球员
        """
        while True:
            self.fsm_state = self.fsm_state_map[self.fsm_state]()
            if self.fsm_state == Tactic21FSM.EXIT:
                break
        # TODO 休整和融合得到的所有结果
        return self.tactic_list

    def seek_kicker(self, index = None): 
        frame_record = self.load_frame_data(index)
        kicker = frame_record["kicker"] if frame_record is not None else None
        return kicker

    def seek_first_kicker(self):
        """
        搜索找到首个踢球者
        """
        self.first_kicker = None
        self.first_kicker_frame_id = None
        kicker = None
        while kicker is None and self.next_frame_id in self.labels_dict:
            kicker = self.seek_kicker(self.next_frame_id)
            self.next_frame_id += 1
        if kicker is not None:
            self.first_kicker = kicker
            self.first_kicker_frame_id = self.next_frame_id - 1
            self.window_cur_index = self.next_frame_id
            return Tactic21FSM.SEEK_SECOND_STAGET_KICKER
        else:
            self.first_kicker = None
            self.first_kicker_frame_id = None
            return Tactic21FSM.EXIT
  
    def seek_second_kicker(self):
        """
        找到第一个踢球者之后去搜索第二个踢球者
        """
        kicker = None
        while kicker is None:
            if self.window_cur_index in self.labels_dict and (self.window_cur_index - self.first_kicker_frame_id) < self.seek_window_size:
                kicker = self.seek_kicker(self.window_cur_index)
                self.window_cur_index += 1
            else:
                break

        if kicker is None:  
            # 窗口内没有找到新的踢球者只能从第一个踢球者重新搜索
            return Tactic21FSM.SEEK_FIRST_STAGET_KICKER


        # 先采用O(kn)的时间复杂度完成搜索 有机会再完成优化
        elif kicker.oid == self.first_kicker.oid and kicker.cls == self.first_kicker.cls:
            # 说明第一个kicker踢球的时间比较长一直在占有球可以适当往后挪一下位置继续搜索即可
            if self.window_cur_index - self.first_kicker_frame_id >= self.seek_window_size // 2:
                self.first_kicker = kicker
                self.first_kicker_frame_id = self.window_cur_index - 1
            return Tactic21FSM.SEEK_SECOND_STAGET_KICKER
        else:
            if (kicker.cls == self.first_kicker.cls) and (kicker.oid != self.first_kicker.oid) and self.get_distance(self.first_kicker, kicker) >= 50:
                # 同一个阵营内则直接将其作为第二个kicker
                # TODO 对于那些不一定非要有回传的情况是否能够判断出来情况更复杂 
                # 那些球场上的边角几乎无法判断 现在的算法只能假设是必须有回传的情况
                self.second_kicker = kicker
                self.second_kicker_frame_id = self.window_cur_index - 1
                return Tactic21FSM.SEEK_THIRD_STAGET_KICKER
            else:
                # 其余的任何条件都不管继续搜索第二个kicker
                return Tactic21FSM.SEEK_SECOND_STAGET_KICKER

    def seek_third_kicker(self):
        """
        搜索第三个kicker
        """
        kicker = None
        while kicker is None and self.window_cur_index in self.labels_dict and (self.window_cur_index - self.second_kicker_frame_id) < self.seek_window_size:
            kicker = self.seek_kicker(self.window_cur_index)
            self.window_cur_index += 1

        if kicker is None:  
            # 窗口内没有找到新的踢球者只能从第一个踢球者重新搜索
            return Tactic21FSM.SEEK_FIRST_STAGET_KICKER
        elif kicker.oid == self.first_kicker.oid and kicker.cls == self.first_kicker.cls:
            # 如果第一个kicker重新出现说明可能已经完成了一个战术动作然后再做进一步判断
            # 判断是否的确越过了某个目标
            # 即判断由第一次kicker 和第二次kicker 以及第三次kicker组成的三个向量 最终是否包围了对方潜在的目标即可
            # 对方的目标是以第一次或者第二次kicker找到的最近的目标
            # 1. 获取kicker2对应的一个记录情况
            kicker1_frame_record = self.load_frame_data(self.first_kicker_frame_id)
            kicker1_kicker2_direction = self.get_direction(self.first_kicker, self.second_kicker)

            # 2. 选择最有威胁的球员 对于每个潜在位置计算对应的对方球员
            # (正好拦在踢球队员的前方道路上且最近 由于都在运动 那么实际上在处理的时候可以做一下权衡 即利用第二个关键点位置时的记录来计算方向向量)
            _, enemy_surroundings = self.get_surroundings(self.first_kicker, kicker1_frame_record)
            front_player = self.get_front_player(self.first_kicker, kicker1_kicker2_direction, enemy_surroundings)
            ret = self.judge_point_positions(self.first_kicker, self.second_kicker, kicker, front_player)
            if ret:
                self.tactic_list.append(Tactic21(self.first_kicker_frame_id, self.window_cur_index - 1, constant.TACTIC_21, self.second_kicker_frame_id, self.first_kicker.oid, self.second_kicker.oid, front_player.oid))
                # print(self.first_kicker.oid, self.second_kicker.oid, front_player.oid, self.first_kicker_frame_id, self.window_cur_index)

            self.next_frame_id = self.window_cur_index
            return Tactic21FSM.SEEK_FIRST_STAGET_KICKER

        else:
            # 继续搜索
            return Tactic21FSM.SEEK_THIRD_STAGET_KICKER

class Tactic32FSM(TacticFSM):
    
    """
    用于战术分析挖掘的有限状态机(仅仅用于2-1战术挖掘)
    """
    SEEK_FIRST_STAGET_KICKER = 0     # 寻找踢球者的第一个阶段 找到一个有效的踢球者即可
    SEEK_SECOND_STAGET_KICKER = 1    # 寻找踢球者的第二个阶段 找到和第一个踢球者同队的配合球员 完成战术的上部分目标搜寻
    SEEK_THIRD_STAGET_KICKER = 2     # 寻找踢球者的第三个阶段 找到第一个球员即可
    SEEK_OTHERS = 3                  # 寻找剩余的球员 直到和前面的球员重复
    EXIT = 4

    def __init__(self) -> None:

        super(Tactic32FSM, self).__init__()

        self.seek_window_size = 100
        
        self.fistr_kicker = None
        self.first_kicker_frame_id = None
        self.second_kicker = None
        self.second_kicker_frame_id = None
        self.third_kicker = None
        self.third_kicker_frame_id = None

        self.first_front_player = None
        self.first_front_player_frame_id = None
        self.second_front_player = None
        self.second_front_player_frame_id = None

        self.remain_kickers = []

        self.tactic_list = []    # 记录找到的所有战术对象
        self.fsm_state = Tactic32FSM.SEEK_FIRST_STAGET_KICKER
        self.fsm_state_map = {
            Tactic32FSM.SEEK_FIRST_STAGET_KICKER: self.seek_first_kicker,
            Tactic32FSM.SEEK_SECOND_STAGET_KICKER: self.seek_second_kicker,
            Tactic32FSM.SEEK_THIRD_STAGET_KICKER: self.seek_third_kicker,
            Tactic32FSM.SEEK_OTHERS: self.seek_others,
        }

    def run(self):
        """
        利用到的主要参数:

        detections: 视频片段中跟踪数据中的所有检测框

        挖掘主流程:

        和2-1基本类似但要首要保证一个二过一
        在找到一个二过一之后 为了简化过程 后面遇到的所有同队球员全部加入候选队列
        直到在一定时间内再重新传回给前面出现过的球员即可(由于不懂足球这样处理显然不怎么准确)
        """
        while True:
            self.fsm_state = self.fsm_state_map[self.fsm_state]()
            if self.fsm_state == Tactic32FSM.EXIT:
                break
        # TODO 休整和融合得到的所有结果
        return self.tactic_list

    def seek_kicker(self, index = None): 
        frame_record = self.load_frame_data(index)
        kicker = frame_record["kicker"] if frame_record is not None else None
        return kicker

    def seek_first_kicker(self):
        """
        搜索找到首个踢球者
        """
        self.first_kicker = None
        self.first_kicker_frame_id = None
        self.second_kicker = None
        self.second_front_player_frame_id = None
        self.third_kicker = None
        self.third_kicker_frame_id = None
        self.first_front_player = None
        self.first_front_player_frame_id = None
        self.second_front_player = None
        self.second_front_player_frame_id = None
        self.remain_kickers = []

        kicker = None
    
        while kicker is None and self.next_frame_id in self.labels_dict:
            kicker = self.seek_kicker(self.next_frame_id)
            self.next_frame_id += 1
        if kicker is not None:
            self.first_kicker = kicker
            self.first_kicker_frame_id = self.next_frame_id - 1
            self.window_cur_index = self.next_frame_id
            return Tactic32FSM.SEEK_SECOND_STAGET_KICKER
        else:
            self.first_kicker = None
            self.first_kicker_frame_id = None
            return Tactic32FSM.EXIT
  
    def seek_second_kicker(self):
        """
        找到第一个踢球者之后去搜索第二个踢球者
        """
        kicker = None
        while kicker is None:
            if self.window_cur_index in self.labels_dict and (self.window_cur_index - self.first_kicker_frame_id) < self.seek_window_size:
                kicker = self.seek_kicker(self.window_cur_index)
                self.window_cur_index += 1
            else:
                break

        if kicker is None:  
            # 窗口内没有找到新的踢球者只能从第一个踢球者重新搜索
            return Tactic32FSM.SEEK_FIRST_STAGET_KICKER


        # 先采用O(kn)的时间复杂度完成搜索 有机会再完成优化
        elif kicker.oid == self.first_kicker.oid:
            # 说明第一个kicker踢球的时间比较长一直在占有球可以适当往后挪一下位置继续搜索即可
            if self.window_cur_index - self.first_kicker_frame_id >= self.seek_window_size // 2:
                self.first_kicker = kicker
                self.first_kicker_frame_id = self.window_cur_index - 1
            return Tactic32FSM.SEEK_SECOND_STAGET_KICKER
        else:
            dist = self.get_distance(self.first_kicker, kicker)
            # 局部战术限制距离范围
            if (kicker.cls == self.first_kicker.cls) and (kicker.oid != self.first_kicker.oid) and dist >= 50 and dist <= 1000:
                # 同一个阵营内则直接将其作为第二个kicker
                # TODO 对于那些不一定非要有回传的情况是否能够判断出来情况更复杂 
                # 那些球场上的边角几乎无法判断 现在的算法只能假设是必须有回传的情况
                self.second_kicker = kicker
                self.second_kicker_frame_id = self.window_cur_index - 1
                return Tactic32FSM.SEEK_THIRD_STAGET_KICKER
            else:
                # 其余的任何条件都不管继续搜索第二个kicker
                return Tactic32FSM.SEEK_SECOND_STAGET_KICKER

    def seek_third_kicker(self):
        """
        搜索第三个kicker
        """
        kicker = None
        while kicker is None and self.window_cur_index in self.labels_dict and (self.window_cur_index - self.second_kicker_frame_id) < self.seek_window_size:
            kicker = self.seek_kicker(self.window_cur_index)
            self.window_cur_index += 1

        if kicker is None:  
            # 窗口内没有找到新的踢球者只能从第一个踢球者重新搜索
            return Tactic32FSM.SEEK_FIRST_STAGET_KICKER
        elif kicker.oid == self.first_kicker.oid:
            # 如果第一个kicker重新出现说明可能已经完成了一个战术动作然后再做进一步判断
            # 判断是否的确越过了某个目标
            # 即判断由第一次kicker 和第二次kicker 以及第三次kicker组成的三个向量 最终是否包围了对方潜在的目标即可
            # 对方的目标是以第一次或者第二次kicker找到的最近的目标
            # 1. 获取kicker2对应的一个记录情况
            # (选择kicker2的原因是正好拦在踢球队员的前方道路上且最近 由于都在运动 那么实际上在处理的时候可以做一下权衡 即利用第二个关键点位置时的记录来计算方向向量)
            kicker2_frame_record = self.load_frame_data(self.second_kicker_frame_id)
            kicker1_kicker2_direction = self.get_direction(self.first_kicker, self.second_kicker)

            # 2. 选择最有威胁的球员 对于每个潜在位置计算对应的对方球员
            _, enemy_surroundings = self.get_surroundings(self.first_kicker, kicker2_frame_record)
            front_player = self.get_front_player(self.first_kicker, kicker1_kicker2_direction, enemy_surroundings)
            # 对运动加一定的缓冲但作用不大
            minx = min([self.first_kicker.xcenter, self.second_kicker.xcenter, kicker.xcenter])
            miny = min([self.first_kicker.ycenter, self.second_kicker.ycenter, kicker.ycenter])
            maxx = max([self.first_kicker.xcenter, self.second_kicker.xcenter, kicker.xcenter])
            maxy = max([self.first_kicker.ycenter, self.second_kicker.ycenter, kicker.ycenter])
            # 不利用外积而是利用范围这个非常粗糙地判断
            if front_player is not None and minx - 15 <= front_player.xcenter <= maxx + 15 and miny - 15 <= front_player.ycenter <= maxy + 15 and self.window_cur_index - self.first_kicker_frame_id >= self.seek_window_size // 2:
                # print("Find it:", self.first_kicker.oid, self.second_kicker.oid, front_player.oid)
                # print(self.first_kicker.oid, self.second_kicker.oid, front_player.oid, self.first_kicker_frame_id, self.window_cur_index)
                self.third_kicker = kicker
                self.third_kicker_frame_id = self.window_cur_index - 1
                self.first_front_player = front_player
                self.first_front_player_frame_id = self.first_kicker_frame_id
                return Tactic32FSM.SEEK_OTHERS

            return Tactic32FSM.SEEK_FIRST_STAGET_KICKER

        else:
            # 继续搜索
            return Tactic32FSM.SEEK_THIRD_STAGET_KICKER

    def seek_others(self):
        """
        当一个三过二的前半部分被满足之后应该考虑剩余的部分(一直往后搜索) (TODO 这里是不怎么合理的 需要判断距离)
        """
        kicker = None
        while kicker is None and self.window_cur_index in self.labels_dict:
            kicker = self.seek_kicker(self.window_cur_index)
            self.window_cur_index += 1

        if kicker is None:
            # 窗口内没有找到新的踢球者只能从第一个踢球者重新搜索
            return Tactic32FSM.SEEK_FIRST_STAGET_KICKER

        if kicker.oid in [self.second_kicker.oid, self.third_kicker.oid]:
            # 防止距离过小而不准确(也可以使用距离判断)
            if self.window_cur_index - self.third_kicker_frame_id > 15:
                # 达到目标
                if len(self.remain_kickers) > 0:  # 必须存在其它的过人情况
                    
                    # print("Tactic 3-2", self.first_kicker.oid, self.second_kicker.oid, self.first_front_player.oid, len(self.remain_kickers))
                    self.remain_kickers.append((kicker, self.window_cur_index - 1))
                    kickers_directions = self.get_direction(self.remain_kickers[0][0], kicker)
                    _, enemy_surroundings = self.get_surroundings(self.remain_kickers[0][0], self.load_frame_data(self.window_cur_index - 1))
                    self.second_front_player = self.get_front_player(self.remain_kickers[0][0], kickers_directions, enemy_surroundings)
                    self.second_front_player_frame_id = self.remain_kickers[0][1]

                    # print(self.second_front_player.oid)
                    
                    # print("Tactic 3-2:")
                    # print("First Kicker:", self.first_kicker.oid, self.first_kicker_frame_id)
                    # print("Second Kicker:", self.second_kicker.oid, self.second_kicker_frame_id)
                    # print("Third Kicker:", self.third_kicker.oid, self.third_kicker_frame_id)
                    # print("First Front:", self.first_front_player.oid, self.first_front_player_frame_id)
                    # print("Second Front:", self.second_front_player.oid, self.second_front_player_frame_id)

                    # for player in self.remain_kickers:
                    #     print("Inner Kicker:", player[0].oid)

                    self.tactic_list.append(
                        Tactic32(
                            self.first_kicker_frame_id,
                            self.window_cur_index - 1,
                            constant.TACTIC_32,
                            self.first_kicker,
                            self.first_kicker_frame_id,
                            self.second_kicker,
                            self.second_kicker_frame_id,
                            self.third_kicker,
                            self.third_kicker_frame_id,
                            self.first_front_player,
                            self.first_front_player_frame_id,
                            self.second_front_player,
                            self.second_front_player_frame_id,
                            self.remain_kickers,
                        )
                    )

                    self.next_frame_id = self.window_cur_index
                    return Tactic32FSM.SEEK_FIRST_STAGET_KICKER

            return Tactic32FSM.SEEK_OTHERS
        else:
            if kicker.cls == self.third_kicker.cls:
                # 同一个传球阵营
                oids = []
                for player in self.remain_kickers:
                    oids.append(player[0].oid)
                if kicker.oid not in oids:
                    self.remain_kickers.append((kicker, self.window_cur_index - 1))
            return Tactic32FSM.SEEK_OTHERS