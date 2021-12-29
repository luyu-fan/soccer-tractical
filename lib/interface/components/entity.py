"""
素材实体
"""

class EntityCard:
    """
    用于视频信息显示的Card组件
    """
    def __init__(
        self,
        root,
        file_name,
        size,
        offset,
        status,
    ):
        """
        Args:
            root: 父组件
            file_name: 绑定的video的名称
            size: 整个标签的大小
            offset: 相对于父组件布局的偏移量
            status: 状态
        """