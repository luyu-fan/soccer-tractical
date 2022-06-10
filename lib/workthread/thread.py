import threading

"""
工作线程
"""
class WorkThread(threading.Thread):
    """
    简单的带有推出信号的工作线程封装
    """
    def __init__(
        self,
        worker_name,
        work_func
    ):
        """
        Args:
            worker_name: 工作线程的名称，不是标识
            work_func: 主要的工作函数
        """
        super(WorkThread, self).__init__()

        self.worker_name = worker_name
        self.work_func = work_func

        # 推出信号
        self.stop_event = threading.Event()
    
    def run(self):
        self.work_func(self.stop_event)

    def stop(self):
        self.stop_event.set()

    def is_stop(self):
        return self.stop_event.is_set()

    def is_alive(self):
        """
        线程是否还处于存活状态
        """
        return super().is_alive()
