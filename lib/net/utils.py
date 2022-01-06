"""
网络相关的辅助功能
"""
import sys, struct

class NetHelper:
    
    def IsLittelEndian():
        endian = sys.byteorder
        if endian == "little":
            return True
        return False

    def Int2Net(num):
        """
        将整数转化为网络字节
        """
        # 转为大端字节 无符号四字节整数
        return struct.pack('>I', num)

    def Net2Int(data_bytes):
        """
        将网络字节转换为整数
        """
        # if NetHelper.IsLittelEndian():
        #     # 转化为16进制小端字节
        #     data_bytes = data_bytes[::-1]

        # 转化为整型 其中byteorder指明此字节数组使用的字节序 可以直接转换
        num = int.from_bytes(data_bytes, byteorder='big', signed=False)
        return num

if __name__ == "__main__":

    num = 1234

    data_bytes = NetHelper.Int2Net(num)
    print(data_bytes, len(data_bytes))
    cvt_num = NetHelper.Net2Int(data_bytes)
    print(cvt_num)
