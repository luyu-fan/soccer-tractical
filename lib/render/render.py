import cv2, numpy as np, random
# from PIL import Image, ImageDraw, ImageFont

from ..utils import bezier_calc as bezier_calc
from ..utils import distance_calc as distance_calc
from ..dataprocess import prepare as prepare

# font_style = ImageFont.truetype("./assets/SimHei.ttf", 12, encoding="utf-8")

def renderArrow(frame, src, dst, color = (32, 64, 96)):
    src_point = (int(src.xcenter), int(src.ycenter))
    dst_point = (int(dst.xcenter + (dst.xcenter - src.xcenter) * 4), int(dst.ycenter + (dst.ycenter - src.ycenter) * 4))
    frame = cv2.arrowedLine(frame, src_point, dst_point, color=color, thickness=2,line_type=cv2.LINE_AA)
    frame = cv2.circle(frame, src_point, radius=8, color=color, thickness=-1)
    return frame

def renderRRectLabel_batch(frame, bbox_records, color = (130, 0, 168), font_color = (255, 255, 255), label_width = 68, label_height = 20):
    """
    在指定的bbox位置上绘制多个圆角矩形标签框 绘制在对bbox的顶部
    Args:
        frame: 需要绘制的帧
        bbox_records: bbox的记录框 一个或多个
        label_width: 标签的像素宽度
        label_height: 标签的像素高度
    Return:
        frame: 修改完毕之后的视频帧
    """
    render_rects_info = []
    for bbox_record in bbox_records:
        l_x1 = int(bbox_record.xcenter - label_width / 2)
        l_y1 = int(bbox_record.ycenter - bbox_record.height / 2 - label_height / 2 - 12)
        top_left = (l_x1, l_y1)
        bottom_right = (l_x1 + label_width, l_y1 + label_height)
        render_rects_info.append((top_left, bottom_right, bbox_record.info))
    
    frame = renderRRect_batch(frame, render_rects_info, 2, color, -1, cv2.LINE_AA)
    frame = renderLabel_batch(frame, render_rects_info, font_color=font_color)

    return frame

def renderRRect_batch(frame, rects_info, radious, color, thickness, linetype):
    """
    在图像中绘制多个圆角矩形
    Args:
        frame: 图像帧
        rects_info: 多个矩形框信息列表
        radious: 圆角半径
        color: 颜色
        thickness: 线的厚度或填充信息
        linetype: 线的类型
    """
    for rect in rects_info:
        renderRRect(frame, rect[0], rect[1], radious, color, thickness, linetype)
    return frame

def renderLabel_batch(frame, rects_info, font_size = 12, font_color = (255,255,255)):
    """
    绘制label标签 由于cv2对unicode字符支持不好 因此只能先利用PIL做一次
    Args:
        frame: 视频帧
        label_center: 标签的中心点
        label_text: 标签文本
    Return:
        frame: 添加了label之后视频帧
    """
    # !!! only use ASCII code to get a fast speed.
    # p_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for rect in rects_info:
        label_center = (rect[0][0] + (rect[1][0] -  rect[0][0]) // 2, rect[0][1] + (rect[1][1] -  rect[0][1]) // 2)
        renderLabel(frame, label_center, rect[2], font_size, font_color)
    return frame
    # return cv2.cvtColor(np.asarray(p_frame), cv2.COLOR_RGB2BGR)

def renderRRect(frame, top_left, bottom_right, radious, color, thickness, linetype):
    """
    在图像中绘制圆角矩形
    Args:
        frame: 图像帧
        top_left: 左上点
        bottom_right: 右下点
        radious: 圆角半径
        color: 颜色
    """
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    # 绘制直线部分
    cv2.line(frame, (top_left[0] + radious, top_left[1]),  (top_right[0] - radious, top_right[1]), color = color, lineType=linetype)
    cv2.line(frame, (top_right[0], top_right[1] + radious),  (bottom_right[0], bottom_right[1] - radious),  color = color, lineType=linetype)
    cv2.line(frame, (bottom_left[0] + radious, bottom_left[1]),  (bottom_right[0] - radious, bottom_right[1]),  color = color, lineType=linetype)
    cv2.line(frame, (top_left[0], top_left[1] + radious),  (bottom_left[0], bottom_left[1] - radious), color = color, lineType=linetype)

    # 绘制中间的矩形
    cv2.rectangle(frame, (top_left[0] + radious, top_left[1] + radious),  (bottom_right[0] - radious, bottom_right[1] - radious), color = color, thickness=thickness, lineType=linetype)

    #  绘制corner部分的椭圆
    cv2.ellipse(frame, (top_left[0] + radious, top_left[1] + radious), (radious, radious), 180.0, 0, 90, color = color, thickness=thickness, lineType=linetype)
    cv2.ellipse(frame, (top_right[0] - radious, top_right[1] + radious), (radious, radious), 270.0, 0, 90, color = color, thickness=thickness, lineType=linetype)
    cv2.ellipse(frame, (bottom_right[0] - radious, bottom_right[1] - radious), (radious, radious), 0.0, 0, 90, color = color, thickness=thickness, lineType=linetype)
    cv2.ellipse(frame, (bottom_left[0] + radious, bottom_left[1] - radious), (radious, radious), 90.0, 0, 90, color = color, thickness=thickness, lineType=linetype)

def renderLabel(frame, label_center, label_text, font_size = 12, font_color = (255,255,255)):
    """
    绘制label标签 由于cv2对unicode字符支持不好 因此只能先利用PIL做一次
    Args:
        frame: 视频帧
        label_center: 标签的中心点
        label_text: 标签文本
    """
    # !!! it has to use the PIL to display the unicode chracters.
    # assert isinstance(frame, Image.Image)
    # p_draw = ImageDraw.Draw(frame)
    # w,h = font_style.getsize(label_text)
    # p_draw.text((label_center[0] - w // 2, label_center[1] - h // 2), label_text, font_color, font_style)
    
    cv2.putText(frame, 
            label_text,
            (label_center[0] - len(label_text) // 2 * 6, label_center[1] + 6),  # fix: align
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=font_color, 
            thickness=1, 
            lineType=cv2.LINE_AA)

def renderTeamShape(frame, convex_points, color, thickness = 1, linetype= cv2.LINE_AA):
    """
    绘制阵型 包括了阵型的凸包和队员的下划线
    Args:
        frame: 图像帧
        convex_points: 凸多边形的顶点
        color: 线的颜色
        thickness: 线的宽度
        linetype: 线的类型
    """
    # convexhull
    if len(convex_points) == 0: return frame
    edges = []
    for point in convex_points:
        edges.append([point.xcenter, point.ycenter + point.height // 2])
    edges = np.asarray([edges], dtype=np.int32)
    cv2.polylines(frame, edges, True, color, thickness=thickness, lineType=linetype)

    # footline
    footLine = []
    for point in convex_points:
        line_points = [[point.xcenter - point.width // 2, point.ycenter + point.height // 2], [point.xcenter + point.width // 2, point.ycenter + point.height // 2]]
        footLine.append(line_points)
    footLine = np.asarray(footLine, dtype=np.int32)
    cv2.polylines(frame, footLine, True, color, thickness = thickness * 4, lineType=linetype)

    return frame

def renderDistance_batch(frame, source, target_bboxes, color, thickness = 1, linetype= cv2.LINE_AA):
    """
    绘制某个目标到另外一个(组)目标之间的连线 以及距离标注(像素)
    Args:
        frame: 帧
        source: 源目标框
        target_bboxes: target_bboxes 需要连接的目标框
        color: 区分的颜色
        thickness: 线的厚度
        linetype: 线的类型
    """
    dist_label_bboxes = []
    for box in target_bboxes:
        if source.oid == box.oid:
            continue
        points = bezier_calc.generate_bezier_points(
            [source.xcenter, source.ycenter - source.height // 2], 
            [(source.xcenter + box.xcenter) / 2, (source.ycenter + box.ycenter) / 2 - 120],
            [box.xcenter, box.ycenter - box.height // 2], 
            int(abs(box.xcenter - source.xcenter)) // 2,
        )
        if len(points) < 1: continue
        fake_bbox = prepare.BBoxRecord(
                "-",
                "-",
                points[len(points) // 2][0],
                points[len(points) // 2][1],
                0,
                0
            )
        fake_bbox.info = str(int(distance_calc.calc_distance_in_pixel((source.xcenter, source.ycenter), (box.xcenter, box.ycenter)))) + " Pixel"
        dist_label_bboxes.append(
            fake_bbox
        )
        dash_line_points = []
        for i in range(len(points)):
            if i % 5 == 4:
                dash_line_points.append([points[i - 1], points[i]])
        dash_line_points = np.asarray(dash_line_points, dtype=np.int32)
        cv2.polylines(frame, dash_line_points, False, color, thickness = thickness * 2, lineType = linetype)
    frame = renderRRectLabel_batch(frame, dist_label_bboxes, color=(127,127,127), font_color=(255,255,255),label_width=160, label_height = 16)
    return frame

def renderBbox_batch(frame, bbox_records):
    """
    绘制对象的bbox
    """
    for bbox in bbox_records:
        x1 = int(bbox.xcenter - bbox.width / 2)
        y1 = int(bbox.ycenter - bbox.height / 2)
        x2 = int(bbox.xcenter + bbox.width / 2)
        y2 = int(bbox.ycenter + bbox.height / 2)
        color = (242, 241, 46)
        cv2.rectangle(frame, (x1, y1),  (x2, y2), color = color, thickness=1, lineType=cv2.LINE_AA)
    return frame