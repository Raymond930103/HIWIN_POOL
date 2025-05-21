
from ..vision.capture import get_next_filename, capture
from ..communicate.tcp_communicate import connect
import os

save_folder = "captured_images"
os.makedirs(save_folder, exist_ok=True)  # 確保資料夾存在
get_next_filename(save_folder)


def arm_capture():
    cap_command = '100'
    while cap_command != "103":
        output = connect(cap_command)
        if output is None:  # 連線失敗則退出
            break
        else:
            print("capture image...")
            capture()
        cap_command = output
        
if __name__ == '__main__':
    arm_capture()