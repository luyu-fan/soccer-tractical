"""
将具体的RGB像素转换为颜色概率
"""

import numpy as np
import math
import cv2

# 论文中提到的11中颜色值 (0 ~ 1 RGB) 
__color_names_mapping = np.array([
    [0, 0, 0],  # black
    [0, 0, 1],  # blue
    [0.5, 0.4, 0.25],  # brown
    [0.5, 0.5, 0.5],   # grey
    [0, 1, 0],         # green
    [1, 0.8, 0],       # orange
    [1, 0.5, 1], # pink
    [1, 0, 1],   # purlple
    [1, 0, 0],   # red
    [1, 1, 1],   # white
    [1, 1, 0],   # yellow
])

# shape 为 (32 * 32 * 32) * 14[r,g,b 11种颜色的概率] 
# "lib/coloring/w2c.txt"
__c2n_table = np.loadtxt("lib/coloring/w2c.txt")[:, 3:]

def get_color_porb(rgb):
    """
    根据单个像素RGB三通道值获取对应的CN
    """
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return __c2n_table[math.floor(r / 8) + 32 * math.floor(g / 8) + 32 * 32 * math.floor(b / 8)]

def get_color_prob_batch(rgbs):
    """
    根据批量像素RGB值获取对应的CN列表
    """
    r = rgbs[:, 0]
    g = rgbs[:, 1]
    b = rgbs[:, 2]
    indexes = (np.floor(r / 8) + 32 * np.floor(g / 8) + 32 * 32 * np.floor(b / 8)).astype(np.uint32)
    return __c2n_table[indexes]


def get_img_mean_rep(img, remove_green = False):
    """
    获得图像的CN均值
    Args:
        img: cv2读取的图像
    """
    img = np.asanyarray(img[:, :, ::-1])
    h, w, c = img.shape
    img = np.reshape(img, (h * w, c))

    rgb_cns = get_color_prob_batch(img)

    if remove_green:
        rgb_cns = np.concatenate((rgb_cns[:, :4], rgb_cns[:, 5:]), axis=1)
    
    mean_rep = np.mean(rgb_cns, axis=0)
    return mean_rep

def image_rerender(img):
    """
    将输入图像按照CN映射重新绘制
    Args:
        img: cv2读取的bgr数据
    """

    # bgr --> rgb --> 3 * (h * w)
    img = np.asanyarray(img[:, :, ::-1])
    h, w, c = img.shape
    img = np.reshape(img, (h * w, c))

    rgb_cns = get_color_prob_batch(img)
    max_index = np.argmax(rgb_cns, axis=1)
    new_img = np.array(
        [
            __color_names_mapping[max_index[i]]  for i in range(max_index.shape[0])
        ]
    ) * 255.0

    new_img = np.reshape(new_img, (h, w, c))
    new_img = np.asanyarray(new_img[:, :, ::-1])
    new_img = new_img.astype(np.uint8)
    return new_img


if __name__ == "__main__":
    
    # rgb = [213,45,240]
    # print(get_color_porb(rgb))

    # rgbs = np.array(
    #     [
    #         [123,34,54],
    #         [200,34,54],
    #         [123,34,226],
    #         [152,34,54],
    #     ]
    # )
    # print(get_color_prob_batch(rgbs))


    img = cv2.imread("car.jpg")
    new_img = image_rerender(img)
    cv2.imshow("newimg", new_img)
    cv2.waitKey(2000)