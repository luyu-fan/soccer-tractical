import os

from . import check_exists, get_relative_data_path
from .video2frames import video2images

def prepare_imgs(videoName):

    # 先检查视频文件是否存在
    videoPath = os.path.join(get_relative_data_path(), "videos", videoName)
    if not check_exists(videoPath):
        raise Exception("Your video file [%s] is not exist in videos folder!".format(videoName))
    else:
        # TODO 检查其它部分
        video2images(videoName)