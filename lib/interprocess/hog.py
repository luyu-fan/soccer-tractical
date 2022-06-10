"""
根据图像标注数据集提取足球HOG特征作为比对库
"""
import cv2

class HOGProcess:
    """
    利用cv2计算图像的HOG特征
    """
    win_size = (16, 16)
    block_size = (8, 8)
    block_stride = (2, 2)
    cell_size = (4, 4)
    n_bins = 9
    hog_descriptor = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)

    locations = ((10, 10), (30, 30), (50, 50), (70, 70),  (90, 90),  (110, 110),  (130, 130),  (150, 150),  (170, 170),  (190, 190))
    win_stride = (2, 2)
    padding = (2, 2)

    fixed_sample_size = (32, 32)

    @staticmethod
    def compute(image):
        """
        计算HOG特征
        """
        # 灰度转换
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, HOGProcess.fixed_sample_size, cv2.INTER_CUBIC)
        hist = HOGProcess.hog_descriptor.compute(gray_image, HOGProcess.win_stride, HOGProcess.padding)
        return hist


