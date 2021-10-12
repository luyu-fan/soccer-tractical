import math

# TODO 添加曼哈顿距离等 判断那种距离最有效

def calc_distance_in_pixel(center1, center2):
    """
    根据中心点计算欧氏像素距离
    Args:
        center1: [x, y]
        center2: [x, y]
    Return:
        dist: float
    """
    return math.sqrt(((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2))
