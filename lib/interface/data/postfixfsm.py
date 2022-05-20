"""
足球简单后置修正有限状态机
"""

from typing import Iterable

class PostFixFSM:
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
        self.window = PostFixWindow(frames_result, window_size, activate_thres)
        self.states_map = {
            PostFixFSM.SEEK: SeekState(),
            PostFixFSM.CANDIDATE: CandidateState(),
            PostFixFSM.FIX: FixState(),
        }

    def run(self):
        """
        FSM后修复检测的主循环
        """
        next_state_name = PostFixFSM.SEEK
        while True:
            cur_state = self.states_map[next_state_name]
            next_state_name = cur_state.do(self.window)
            if next_state_name == PostFixFSM.FINISH:
                return

class PostFixWindow:

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
        assert isinstance(window, PostFixWindow)

        # 找到第一个非空检测即可
        probe_index = window.get_start_index()
        while probe_index < window.get_total_result_num():
            det =  window.get_det(probe_index)
            if det is not None:
                window.move(probe_index - window.get_start_index())
                return PostFixFSM.CANDIDATE
            else:
                probe_index  += 1
        
        return PostFixFSM.FINISH

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
        assert isinstance(window, PostFixWindow)

        # 窗口中实际容量仅为1
        if window.get_start_index() >= window.get_total_result_num() - 1:
            # 忽略掉直接返回完成状态
            # TODO 应该考虑更详细的边界情况
            return PostFixFSM.FINISH

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
            return PostFixFSM.FIX
        else:
            window.discard_det(window.get_start_index())
            return PostFixFSM.SEEK

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
        assert isinstance(window, PostFixWindow)

        start_index = window.get_start_index()
        end = min(start_index + window.size, window.total_num)
        for k in range(start_index, end):
            # 对第k个进行尝试性修复
            window.fix(k)
            if window.fixed_index < k:
                window.fixed_index = k
                
        # 移动窗口
        window.move(window.size // 2 + 1)
        return PostFixFSM.SEEK
 