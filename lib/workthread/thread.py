import threading

"""
工作线程
"""

class WorkThread(threading.Thread):

    def __init__(self, worker_name, work_func):
        super(WorkThread, self).__init__()

        # threading.Thread.__init__(self)
        self.worker_name = worker_name
        self.work_func = work_func
    
    def run(self):
        self.work_func()
