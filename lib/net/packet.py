"""
传输需要的序列化和反序列化封装类
"""
import pickle

from .utils import NetHelper

# | 定界符# | 图像数据的字节总长度(4字节) 包括帧序号所占的四字节 | 帧序号(4字节) | 3通道图像字节 | 
class FrameDataPacket:

    def __init__(self):
        pass

    def encap(
        self,
        frame_id,
        frame,
    ):
        """
        Args:
            frame_id: 帧序号
            frame: 图像帧
        """
        self.frame_id = frame_id
        self.frame = frame

    def get_frame_id(self):
        return self.frame_id
    
    def get_frame(self):
        return self.frame

    def decode(self, data_bytes):
        """
        将二进制数据转换为frame_id和frame
        """
        self.frame_id = NetHelper.Net2Int(data_bytes[:4])
        self.frame = pickle.loads(data_bytes[4:], encoding="ASCII")

    def encode(self):
        """
        将帧id和帧数据转换为bytes数组
        """
        frame_id_bytes = NetHelper.Int2Net(self.frame_id)
        frame_bytes = pickle.dumps(self.frame)
        return (frame_id_bytes + frame_bytes)

# | 定界符# | 数据的字节总长度(4字节) | 结果数据 | 
class MOTResultPacket:
    """
    经过检测处理之后的结果数据Packet
    """
    def __init__(
        self,
        player_result = None,
        ball_result = None,
    ):

        self.result = {
            "player": player_result,
            "ball": ball_result,
        }
    
    def encode(self):
        return pickle.dumps(self.result)

    def decode(self, data_bytes):
        self.result = pickle.loads(data_bytes, encoding="ASCII")
        return self.result

if __name__ == "__main__":
    
    import cv2
    frame = cv2.imread("../example.jpg")
    frame_id = 1
    frame_pkt = FrameDataPacket()
    frame_pkt.encap(frame_id, frame)
    
    data_bytes = frame_pkt.encode()

    print(len(data_bytes))

    new_frame_pkt = FrameDataPacket()
    new_frame_pkt.decode(data_bytes)
    cv2.imshow("frame", new_frame_pkt.get_frame())
    cv2.waitKey(2000)
    