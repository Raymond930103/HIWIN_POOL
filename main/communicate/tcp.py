import socket
import time
from typing import Optional


def create_connection(host: str,
                      port: int,
                      max_retries: int = 3,
                      retry_delay: int = 5) -> Optional[socket.socket]:
    """
    嘗試建立 TCP 連線並內建重試機制。
    連線成功時回傳已連線的 socket；失敗則回傳 None。
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"嘗試連線到 {host}:{port}... (第 {attempt} 次)")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print("已連線成功！")
            return sock         # 連線成功就直接回傳
        except Exception as e:
            print(f"連線失敗: {e}")
            if attempt < max_retries:
                print(f"等待 {retry_delay} 秒後再嘗試連線...")
                time.sleep(retry_delay)
            else:
                print("已達最大重試次數，放棄連線。")
    return None


def send_message(sock: socket.socket, payload: str) -> None:
    """
    將字串包裝成 {payload} 格式後送出。
    呼叫者需保證 sock 已連線且仍然有效。
    """
    message = f"{{{payload}}}"
    sock.sendall(message.encode("utf-8"))
    print(f"已送出訊息：{message}")


def receive_message(sock: socket.socket,
                    bufsize: int = 1024,
                    strip_braces: bool = True) -> Optional[str]:
    """
    等待並接收伺服器回傳資料。
    回傳解碼後的字串，若 strip_braces 為 True 則移除前後大括號。
    若連線被關閉或無資料則回傳 None。
    """
    data = sock.recv(bufsize)
    if not data:
        #print("未收到任何回應。")
        return None

    text = data.decode("utf-8", errors="replace")
    if strip_braces and text.startswith("{") and text.endswith("}"):
        text = text[1:-1]      # 去除首尾花括號
    print(f"伺服器回應：{text}")
    return text


if __name__ == "__main__":
    import time
    HOST = "172.26.155.219"
    PORT = 4000
    

    # 1️⃣ 建立連線（含重試）
    sock = create_connection(HOST, PORT)
    while True:
        if sock is None:
            print("無法建立連線，程式結束。")
            exit(1)
        else:
            send_message(sock, "0")  # 傳送測試訊息
            time.sleep(3)
    

