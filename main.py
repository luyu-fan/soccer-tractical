import cv2

from dataprocess import prepare
from render import render
from interaction import interaction
from utils import team_shape

def app(videoName):
    """
    应用入口
    """
    KICK_DIST_PIXEL_THRES = 60
    SURROUNDING_MAX_DIST_THRES = 400
    PROBE_TTL = 60

    prepare.prepare_frames(videoName)
    labels_dict = prepare.prepare_labels(videoName, kick_dist_pixel_thres=KICK_DIST_PIXEL_THRES)

    probe_kicker_cls = ""
    probe_kicker_oid = ""
    probe_kicker_up_frame_num = 0
    frame_num = 1
    while frame_num in labels_dict.keys():
        frame_record = labels_dict[frame_num]
        img_path = "datasets/images/{:s}/{:06d}.jpg".format(videoName.split(".")[0],frame_num)
        frame = cv2.imread(img_path)
        ball = frame_record["ball"]
        cur_kicker = frame_record["kicker"]
        if ball is not None:
            # print("==>", frame_record["ball"].xcenter, frame_record["ball"].ycenter)
            # 1. 将识别到的足球给绘制出来. 标明位置
            frame = render.renderRRectLabel_batch(frame, [ball], color=(36,36,36))
            if cur_kicker is not None:
                # 3. 将当前帧kicker的周围按照范围将所有的对象检测出来 绘制战术进攻阵型或者防守阵型 显然这里速度很慢 需要批处理 可以看作是一个凸包
                surroundings = interaction.find_surroundings(cur_kicker,frame_record["bbox"], surrounding_max_dist_thres=SURROUNDING_MAX_DIST_THRES)
                self_team_shape = team_shape.convexhull_calc(surroundings[0])
                enemy_team_shape = team_shape.convexhull_calc(surroundings[1])
                frame = render.renderTeamShape(frame,self_team_shape,(146,224,186))
                frame = render.renderTeamShape(frame,enemy_team_shape,(224,186,146))
                frame = render.renderRRectLabel_batch(frame, self_team_shape, (242, 168, 123))
                frame = render.renderRRectLabel_batch(frame, enemy_team_shape, (48, 96, 166))
                # 4. 绘制当前kicker到其它队友或者是地方的一个距离 绘制曲线
                frame = render.renderDistance_batch(frame, cur_kicker, self_team_shape, color=(16,255,16))
                frame = render.renderDistance_batch(frame, cur_kicker, enemy_team_shape, color=(16,16,255))

                # 2. 如果当前帧存在kicker 则将当前帧的kicker给绘制出来
                frame = render.renderRRectLabel_batch(frame, [cur_kicker], color=(255, 255, 255), font_color=(0, 0, 0), label_width=96, label_height=30)

                # 5. 在ttl帧数窗口内探测下一个kicker
                if probe_kicker_up_frame_num > frame_num:
                    for bbox in frame_record["bbox"]:
                        if (probe_kicker_oid == bbox.oid and probe_kicker_cls == bbox.cls):
                            frame = render.renderRRectLabel_batch(frame, [bbox], color=(0, 0, 255), font_color=(0, 0, 0), label_width=96, label_height=30)
                            break
                else:
                    frame_probe_num = frame_num + 1
                    probe_ttl = PROBE_TTL
                    while probe_ttl > 0 and (frame_probe_num in labels_dict.keys()):
                        probe_kicker = labels_dict[frame_probe_num]["kicker"]
                        # 这里只是为了修复自我标注时相同的id 如果采用检测的结果就不会有这种问题
                        if probe_kicker is not None and (probe_kicker.oid != cur_kicker.oid or (probe_kicker.oid == cur_kicker.oid and probe_kicker.cls != cur_kicker.cls)):
                            probe_kicker_up_frame_num = frame_probe_num
                            probe_kicker_cls = probe_kicker.cls
                            probe_kicker_oid = probe_kicker.oid
                            break
                        frame_probe_num += 1
                        probe_ttl -= 1
        cv2.imshow("SoccerFrame", frame)
        cv2.waitKey(5)
        frame_num += 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # NOSV9C_37.txt
    # BXZNP1_17.mp4
    app("BXZNP1_17.mp4")