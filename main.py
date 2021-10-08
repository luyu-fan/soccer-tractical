import cv2

from dataprocess import prepare
from render import render
from interaction import interaction
from utils import team_shape

def app():
    """
    应用入口
    """
    prepare.prepare_frames("BXZNP1_17.mp4")
    labels_dict = prepare.prepare_labels("BXZNP1_17.mp4", kick_dist_pixel_thres=50)

    frame_num = 1
    while frame_num in labels_dict.keys():
        frame_record = labels_dict[frame_num]
        frame = cv2.imread("datasets/images/BXZNP1_17/{:06d}.jpg".format(frame_num))
        if frame_record["ball"] is not None:
            # print("==>", frame_record["ball"].xcenter, frame_record["ball"].ycenter)
            # 1. 将识别到的足球给绘制出来. 标明位置
            frame = render.renderRRectLabel_batch(frame, [frame_record["ball"]])
            if frame_record["kicker"] is not None:
                # 2. 如果当前帧存在kicker 则将当前帧的kicker给绘制出来
                frame = render.renderRRectLabel_batch(frame, [frame_record["kicker"]])
                # 3. 将当前帧kicker的周围按照范围将所有的对象检测出来 绘制战术进攻阵型或者防守阵型 显然这里速度很慢 需要批处理 可以看作是一个凸包
                surroundings = interaction.find_surroundings(frame_record["kicker"],frame_record["bbox"], surrounding_max_dist_thres=250)
                self_team_shape = team_shape.convexhull_calc(surroundings[0])
                enemy_team_shape = team_shape.convexhull_calc(surroundings[1])
                render.renderTeamShape(frame,self_team_shape,(146,224,186))
                render.renderTeamShape(frame,enemy_team_shape,(224,186,146))
                pass
        cv2.imshow("SoccerFrame", frame)
        cv2.waitKey(1)
        frame_num += 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app()
