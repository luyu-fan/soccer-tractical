"""
处理各个控件公共行为的slots中心 是一个不完全不规范的pub-sub模式
"""
class SlotsHub:
    """
    跨控件事件处理代理中心
    """
    __hub__ = {}

    def register(self, event_name, callfn):
        """
        Args:
            event_name: 事件名称
            callfn: 事件处理函数
        """
        SlotsHub.__hub__[event_name] = callfn

    def get_handler(self, event_name):
        """
        Args:
            event_name: 事件名称
        """
        if event_name not in SlotsHub.__hub__:
            print(event_name, "is not registered.")
            raise KeyError
        return SlotsHub.__hub__[event_name]
    
