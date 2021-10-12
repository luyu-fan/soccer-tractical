"""
根据多组点 计算这组点的外接凸包
"""
import numpy as np
from scipy.spatial import ConvexHull

def convexhull_calc(bboxes):
    """
    Args:
        bboxes: 一组对象的bboxes
    Return:
        convex_points: 各个顶点信息
    """
    convex_points = []
    bboxes_num = len(bboxes)
    if bboxes_num <= 1: 
        pass
    elif bboxes_num == 2:
        convex_points = [bboxes[0], bboxes[1]]
    else:
        points = []
        for bbox in bboxes:
            points.append([bbox.xcenter, bbox.ycenter])
        hull = ConvexHull(np.asarray(points))
        convex_points = [bboxes[0]]
        for simplex in hull.simplices:
            convex_points.append(bboxes[simplex[1]])
    return convex_points
    
