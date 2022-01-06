"""
简单的客户端代码
"""
import socket
import cv2

from .utils import NetHelper
from .packet import FrameDataPacket, MOTResultPacket

class MOTClient:

    def __init__(
        self,
        remote_ip,
        remote_port,
    ):
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.conn = socket.socket()

    def connect(self):
        """
        连接
        """
        try:
            self.conn.connect((self.remote_ip, self.remote_port))
        except Exception as e:
            print("Info:", e)
            raise e

    def find_delimiter(self, delimiter = b'#'):
        """
        找到帧边界
        """
        while True:
            try:
                delimiter = self.conn.recv(1)
            except Exception as e:
                print("Info:", e)
                raise e
            else:
                if delimiter == b'#':
                    break

    def get_total_length(self):
        """
        读取承载数据的总长度
        """
        total_length_bytes = self.conn.recv(4)
        total_length = NetHelper.Net2Int(total_length_bytes)
        return total_length

    def get_payload_data(self, total_length):
        """
        解决半包问题
        """
        cur_read_length = 0
        next_read_length = total_length
        recv_pkt = b''
        while cur_read_length != total_length:
            cur_data = self.conn.recv(next_read_length)
            cur_read_length += len(cur_data)
            next_read_length -= len(cur_data)
            recv_pkt += cur_data
        return recv_pkt

    def recv(self):
        """
        读取结果
        """
        # 1. 找到帧边界
        self.find_delimiter(b'#')

        # 2. 读取数据总长度
        total_length = self.get_total_length()
        
        # 3. 读取跟踪数据
        mot_pkt = self.get_payload_data(total_length)

        # 4. 获取mot数据
        mot_result = MOTResultPacket().decode(mot_pkt)
        return mot_result

    def send(self, frame_id, frame):
        """
        Args:
            frame_id: 帧序号
            frame: 图像数据
        """
        pkt = FrameDataPacket()
        pkt.encap(frame_id, frame)
        data_bytes = pkt.encode()

        pkt_data = b'#' + NetHelper.Int2Net(len(data_bytes)) + data_bytes

        try:
            self.conn.sendall(pkt_data)
        except Exception as e:
            print("Info:", e)
            raise e

    def close(self):
        self.conn.close()

if __name__ == "__main__":

    client = MOTClient("127.0.0.1", 9900)

    img = cv2.imread("example.jpg")
    print(type(img))

    client.connect()


