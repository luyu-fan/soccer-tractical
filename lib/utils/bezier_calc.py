import bezier
import numpy as np

# !!! 注意 bezier中会有_speedup加速，由于本机不支持暂时将其注释掉了

def generate_bezier_points(start_point, control_point, end_point, sample_num):
    """
    根据控制点生成贝塞尔曲线
    Args:
        start_point: 起始点
        control_point: 控制点
        end_point: 结束点
        sample_num: 采样个数
    Return:
        points: 采样点的坐标
    """
    # !!! normalize 否则会计算出错 即将值统一规范化到0和1之间然后再乘回去
    max_value = max(start_point[0], start_point[1], end_point[0], end_point[1])
    nodes = np.asfortranarray([
        [start_point[0] / max_value, control_point[0] / max_value, end_point[0] / max_value],
        [start_point[1] / max_value, control_point[1] / max_value, end_point[1] / max_value],
    ])
    curve = bezier.Curve(nodes, degree=2)
    sample_points = np.linspace(0.0, 1.0, sample_num)
    evaulate_points = curve.evaluate_multi(sample_points)
    points = []
    for i in range(0, len(evaulate_points[0])):
        points.append([int(evaulate_points[0][i] * max_value), int(evaulate_points[1][i] * max_value)])
    return points
    