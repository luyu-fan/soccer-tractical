#!/bin/bash

# 1. 准备环境
echo "creating python env......"
conda deactivate
conda create -n soccer-tracking python==3.8.0 -y
conda activate soccer-tracking
conda_home=$(conda info | grep -i "base env")
conda_info=(${conda_home[@]})
conda_root=${conda_info[3]}
echo $conda_root
source $conda_root"/etc/profile.d/conda.sh"
conda activate soccer-tracking
echo "env prepared!"

# 2. 更新项目所在的根目录到代码中以便于获取视频图像渲染
echo "" >> ./lib/constant/constant.py
echo "DATA_ROOT=r'$(pwd)/datasets'" >> ./lib/constant/constant.py   # 如果不支持shell则直接替换路径到对应的源文件即可
echo "hint: if you want to communicate with your ai server, you should modify the endpoint at ./lib/constant/constant.py"


# 3. 安装各类包
conda activate soccer-tracking
echo "Installing some packages in order..."
pip install ttkthemes -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sklearn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install bezier -i https://pypi.tuna.tsinghua.edu.cn/simple


# 4. 运行
echo "============================================================================================="
echo "Now you can run the command 'python main.py' in the 'soccer-tracking' env at the next time."
echo "hint: make sure you have changed the env to 'soccer-tracking'"
echo "hint: just run 'python main.py'"

python main.py





