from vision.yoloball import capture_balls
from communicate.tcp import create_connection, send_message, receive_message
import socket
from configs.setting import HOST, PORT
from run_shot import plan_shot_from_json

import time
import math 

j6_diff = 4


if __name__ == "__main__":
    CORD_JSON = "/Users/caiminhan/Projects/HIWIN_MAIN/captures_json/cords.json"

    dots = 0          # loading 點數
    sock = None
    while True:
        # ----------- 確保連線 -----------
        if sock is None:
            sock = create_connection(HOST, PORT)
            if sock is None:                  # 建立失敗，5 秒後重試
                time.sleep(5)
                continue
            sock.settimeout(2)                # 2 秒沒資料就丟 socket.timeout

        # ----------- 嘗試收訊息 -----------
        try:
            msg = receive_message(sock)
            if msg is None:                   # 沒資料 → 顯示 loading
                loading = "." * dots
                print(f"\r等待中{loading:<3}", end="", flush=True)
                dots = (dots + 1) % 4
                continue

            print(f"\n收到伺服器訊息：{msg}")   # 有資料就換行顯示

            # ----------- 指令判斷 -----------
            if msg == "MOVING":
                print("開始拍攝")
                capture_balls(wait_sec=3, show=False, intrinsics_path="/Users/caiminhan/Projects/HIWIN_MAIN/main/vision/intrinsics.yaml")
                result = plan_shot_from_json(CORD_JSON, 'min', show=False)
                
                if result is None:
                    send_message(sock, "200") # 無法計算路徑
                else:
                    send_message(sock, "100") # 成功計算路徑
                    angle, cue_xy = result
                    print(f"計算結果：{angle:.2f}°，{cue_xy}")
                    arm_angle = -angle                       # 依你原本邏輯
                    arm_x = round(cue_xy[0] * 1000, 2)       # m → mm
                    arm_y = round(375 - cue_xy[1] * 1000, 2) # 底邊座標換算
                    '''
                    dx = j6_diff * math.cos(math.radians(arm_angle))
                    dy = j6_diff * math.sin(math.radians(arm_angle))
                    
                    arm_x += dx
                    arm_y += dy
                    '''
                    payload = f"{arm_angle:.2f}, {arm_x:.2f}, {arm_y:.2f}"
                    send_message(sock, payload)
                    
            elif msg == "EXIT":                          # 伺服器要求關閉
                print("伺服器結束連線，5 秒後嘗試重新連線")
                sock.close()
                sock = None
                time.sleep(5)

            else:
                # 其他訊息可在這裡加 elif 處理
                pass

        # ----------- 例外處理 -----------
        except socket.timeout:
            # 超時等同沒資料，做 loading 動畫
            loading = "." * dots
            print(f"\r等待中{loading:<3}", end="", flush=True)
            dots = (dots + 1) % 4

        except Exception as e:
            print(f"\n連線異常：{e}，5 秒後重試")
            if sock:
                sock.close()
            sock = None
            time.sleep(5)
