from vision.yoloball import capture_balls
from communicate.tcp import create_connection, send_message, receive_message
from configs.setting import HOST, PORT
from run_shot import plan_shot_from_json, get_last_plan_shot_error
from configs.table import TABLE_H_CM
from configs.correction import apply_fudge

import time


def compute_payload_from_latest_json(json_path: str):
    result = plan_shot_from_json(json_path, 'min', show=False)
    if result is None:
        err = get_last_plan_shot_error()
        if err:
            print(f"plan_shot 失敗原因：\n{err}")
        return None
    angle, cue_xy = result
    # cue_xy is in meters → convert to cm for fudge, then to mm for robot
    x_cm, y_cm = cue_xy[0] * 100.0, cue_xy[1] * 100.0
    x_cm, y_cm = apply_fudge(x_cm, y_cm)
    arm_angle = -angle
    arm_x = round(x_cm * 10.0, 2)
    arm_y = round(TABLE_H_CM * 10.0 - y_cm * 10.0, 2)
    return f"{arm_angle:.2f}, {arm_x:.2f}, {arm_y:.2f}", result


if __name__ == "__main__":
    INTRINSICS = "/Users/caiminhan/Projects/HIWIN_MAIN/main/vision/intrinsics.yaml"
    while True:
        # 1) Connect to robot server
        sock = create_connection(HOST, PORT)
        if sock is None:
            print("無法建立連線，5 秒後重試。")
            time.sleep(5)
            continue
        print("已連線，等待 CONNECTED/MOVING 訊息…")

        # 2) Expect CONNECTED then MOVING
        try:
            # Block until CONNECTED/MOVING
            while True:
                msg = receive_message(sock)
                if msg is None:
                    raise RuntimeError("連線已關閉")
                print(f"收到伺服器訊息：{msg}")
                if msg == "CONNECTED":
                    continue
                if msg == "MOVING":
                    break

            # 3) Capture and compute once MOVING arrives
            print("開始拍攝與計算…")
            json_path, _ = capture_balls(wait_sec=3, show=False, intrinsics_path=INTRINSICS)
            plan_error = None
            payload = None
            payload_and_result = compute_payload_from_latest_json(json_path) if json_path else None
            if payload_and_result is not None:
                payload, (angle, cue_xy) = payload_and_result
                print(f"計算結果：{angle:.2f}°，{cue_xy}")
            else:
                plan_error = get_last_plan_shot_error() if json_path else "無法取得球桌影像，拍攝失敗"
                plan_error = plan_error or "路徑規劃失敗，請重試"
                print(f"路徑規劃失敗：{plan_error}")

            # 4) Send '100', wait for 'wait', then send coordinates
            if payload is not None:
                send_message(sock, "100")
                while True:
                    msg = receive_message(sock)
                    if msg is None:
                        raise RuntimeError("等待 wait 訊息時連線關閉")
                    print(f"收到伺服器訊息：{msg}")
                    if msg.strip().lower() == "wait":
                        break
                send_message(sock, payload)
                print(f"已送出座標：{payload}")
            else:
                send_message(sock, "200")
                print("已通知機械手臂：本次規劃失敗 (代碼 200)")

            # 5) Wait for DONE then loop (server will close connection)
            while True:
                msg = receive_message(sock)
                if msg is None:
                    print("連線關閉，重新連線中…")
                    break
                print(f"收到伺服器訊息：{msg}")
                if msg == "DONE":
                    print("動作完成，關閉連線並重新開始…")
                    break

        except Exception as e:
            print(f"連線或通訊異常：{e}")
        finally:
            try:
                sock.close()
            except Exception:
                pass
            time.sleep(3)
