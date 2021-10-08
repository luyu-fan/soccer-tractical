"""
根据bbox进行一些计算
"""

from utils.distance_calc import calc_distance_in_pixel

def find_surroundings(target, bboxes, surrounding_min_dist_thres = 0, surrounding_max_dist_thres = 200):
    """
    找到目标一定范围内的潜在交互对象
    Args:
        target: 目标
        bboxes: 当前帧的所有对象信息
        surrounding_min_dist_thres: 最小距离阈值
        surrounding_max_dist_thres: 最大距离阈值
    Return:
        potential_bboxes: 过滤之后的潜在交互bbox元组,([],[]) 按照身份分类 第一个是和当前身份相同的 第二个是不同的
    """
    potential_role_bboxes = ([], [])
    for box in bboxes:
        if box.cls == "Ball":
            continue
        dist = calc_distance_in_pixel((target.xcenter, target.ycenter), (box.xcenter, box.ycenter))
        if dist >= surrounding_min_dist_thres and dist <= surrounding_max_dist_thres:
            if box.cls == target.cls:
                potential_role_bboxes[0].append(box)
            else:
                potential_role_bboxes[1].append(box)
    return potential_role_bboxes