"""
存放一些全局常量
"""
# 此处为了方便调试将服务端信息硬编码 在后续完善环节应该将其通过配置文件的形式读入
REMOTE_IP = "127.0.0.1"
REMOTE_PORT = 9900

SWITCH_FRAME_EVENT = "Switch_Frame_Event"
FINISHED_VIDEOS = "Finished_Videos"
PROCESSING_VIDEOS = "Processing_Videos"
VIDEOS = "Videos"
TACTICS = "Tactics"

TACTIC_21 = "2-1"
TACTIC_32 = "3-2"

SWITCH_WELCOME_FRAME_CODE = 0
SWITCH_LIBRARY_FRAME_CODE = 1
SWITCH_PLAYER_FRAME_CODE = 2
SWITCH_TACTICS_FRAME_CODE = 3

SHALLOW_FRAME_BACKGROUND_NAME = "Shallow.TFrame"
DARK_FRAME_BACKGROUND_NAME = "Dark.TFrame"

TITLE_TEXT_STYLE_NAME = "Title.TLabel"
FRAME_SEPARATOR_LINE_NAME = "Separator.TFrame"

DESC_TEXT_STYLE_NAME = "Desc.TLabel"
TIP_TEXT_STYLE_NAME = "Tip.TLabel"

DARK_BTN_BACKGROUND_NAME = "Dark.TButton"
SHALLOW_BTN_BACKGROUND_NAME = "Shallow.TButton"
WEL_ENTER_BTN = "Whilte.TButton"

# 程序存放视频已经中间处理数据的绝对路径
DATA_ROOT = r"C:\Users\luyu-fan\Desktop\mdp\main-code\fake-tractical\datasets"