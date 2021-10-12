# 获得当前目录相对于数据目录的路径
import os

def get_relative_data_path():
    return "./datasets/"

def check_exists(folder):
    return os.path.exists(folder)

def make_folder(folder):
    os.mkdir(folder)