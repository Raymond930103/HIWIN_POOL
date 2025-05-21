import socket
import time


def connect(HOST, PORT, input):
    
    MAX_RETRIES = 3         # 最多重試次數
    RETRY_DELAY = 5         # 每次重試間隔 (秒)
    
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                print(f"嘗試連線到 {HOST}:{PORT}... (第 {attempt+1} 次)")
                s.connect((HOST, PORT))
                print("已連線成功！")

                # 傳送測試訊息 (根據實際需求修改訊息或指令格式)
                message =  f"{{{input}}}"
                s.sendall(message.encode('utf-8'))
                print(f"已送出訊息：{message.strip()}")

                # 等待伺服器回應，最多接收 1024 bytes
                data = s.recv(1024)
                output = data.decode('utf-8', errors='replace').strip("{}")
                if data:
                    print(f"伺服器回應：{output}")
                    return output
                else:
                    print("未收到任何回應。")

                # 若執行到此，表示連線成功且已送出/接收完畢，故可直接跳出 while 迴圈
                break

        except Exception as e:
            print(f"連線失敗: {e}")
            attempt += 1
            if attempt < MAX_RETRIES:
                print(f"等待 {RETRY_DELAY} 秒後再嘗試連線...")
                time.sleep(RETRY_DELAY)
            else:
                print("已達最大重試次數，程式結束。")
                
                
if __name__ == '__main__':
    connect(HOST = '192.168.0.152',PORT = 4000,input = '')  # 傳送 '100' 給伺服器
    