"""
程序运行时涉及到的所有中间数据的集合
"""
class DataHub:

    """
    存放和管理程序所需所有数据的静态类
    """
    __global_data_hub = {}

    def __init__(self) -> None:
        ...

    def get(self, data_name):
        """
        根据数据名称返回对应的数据内容
        Args:
            data_name: 数据名称
        """
        if data_name not in DataHub.__global_data_hub.keys():
            raise KeyError
        return DataHub.__global_data_hub[data_name]

    def set(self, data_name, data):
        """
        设置对应的数据。当前简单版本会覆盖原来的数据。
        Args:
            data_name: 数据名称
            data: 对应的数据集合
        """
        DataHub.__global_data_hub[data_name] = data

    
    