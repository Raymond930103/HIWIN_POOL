import os
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Make project root importable
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.vision.yoloball import capture_balls
from main.run_shot import plan_shot_from_json, get_last_plan_shot_error
from main.configs.setting import HOST, PORT
from main.communicate.tcp import create_connection, send_message, receive_message
from .config import Config
from main.configs.table import TABLE_H_CM
from main.configs.correction import apply_fudge


def compute_arm_payload(angle_deg: float, cue_xy_m: Tuple[float, float]) -> str:
    # Convert to cm for fudge correction, then to mm for robot
    x_cm, y_cm = cue_xy_m[0] * 100.0, cue_xy_m[1] * 100.0
    x_cm, y_cm = apply_fudge(x_cm, y_cm)
    arm_angle = -angle_deg
    arm_x = round(x_cm * 10.0, 2)  # cm → mm
    arm_y = round(TABLE_H_CM * 10.0 - y_cm * 10.0, 2)  # bottom-origin conversion
    return f"{arm_angle:.2f}, {arm_x:.2f}, {arm_y:.2f}"


def send_to_robot(payload: str, *, wait_reply: bool = True, timeout: float = 5.0, send_ok_first: bool = True):
    """Deprecated simple sender preserved for compatibility.
    Prefer `perform_handshaked_strike` which follows CONNECTED→MOVING→100+coords→DONE.
    """
    sock = None
    try:
        sock = create_connection(HOST, PORT)
        if sock is None:
            return False, None
        if wait_reply:
            try:
                sock.settimeout(timeout)
            except Exception:
                pass
        if send_ok_first:
            send_message(sock, "100")
        send_message(sock, payload)
        reply = None
        if wait_reply:
            try:
                reply = receive_message(sock)
            except Exception:
                reply = None
        return True, reply
    except Exception:
        return False, None
    finally:
        try:
            if sock:
                sock.close()
        except Exception:
            pass


def perform_handshaked_strike(*,
                              target: str = "min",
                              intrinsics_path: Optional[str] = None,
                              wait_moving_max: Optional[float] = None,   # None = wait indefinitely
                              wait_done_max: Optional[float] = 180.0,
                              recv_poll_timeout: float = 1.0):
    """Full session following robot server protocol:
    1) Connect; expect 'CONNECTED'.
    2) Wait for 'MOVING'.
    3) Run capture+plan; compute payload.
    4) Send '100' (success) or '200' (failure). When success wait for 'wait' then send payload.
    5) Wait for 'DONE' and close.

    Returns (balls_data, result, payload, reply, error_msg)
    - result is Optional[Tuple[angle_deg, (cue_x_m, cue_y_m)]]
    - reply is final message received (e.g., 'DONE') or None
    - error_msg is a human-friendly reason suitable for UI display
    """
    sock = None
    balls_data = {"timestamp": datetime.utcnow().isoformat(), "balls": []}
    result = None
    payload = None
    reply = None
    error_msg = None

    try:
        sock = create_connection(HOST, PORT)
        if sock is None:
            error_msg = "無法建立與機械手臂的連線"
            return balls_data, result, payload, reply, error_msg
        try:
            sock.settimeout(recv_poll_timeout)
        except Exception:
            pass

        # 1/2) Wait for CONNECTED then MOVING (MOVING may take long until human presses button)
        start = time.monotonic()
        while True:
            try:
                msg = receive_message(sock)
            except Exception:
                msg = None

            if msg:
                if msg == "CONNECTED":
                    # proceed to wait MOVING
                    continue
                if msg == "MOVING":
                    break
                # ignore any other noise

            if wait_moving_max is not None and (time.monotonic() - start) > wait_moving_max:
                reply = "timeout_waiting_MOVING"
                error_msg = "等待機械手臂 MOVING 訊息逾時"
                return balls_data, result, payload, reply, error_msg

        # 3) Capture and plan now that robot is moving
        json_path, data = capture_balls(
            wait_sec=3,
            show=False,
            intrinsics_path=intrinsics_path,
            preview=False,
            cam_index=getattr(Config, 'CAMERA_INDEX', 0),
            cam_width=getattr(Config, 'CAMERA_WIDTH', None),
            cam_height=getattr(Config, 'CAMERA_HEIGHT', None),
        )
        if data:
            balls_data = data
        if json_path:
            result = plan_shot_from_json(json_path, target, show=False)

        # Compute payload or handle no-plan case
        if result is not None:
            angle_deg, cue_xy = result
            payload = compute_arm_payload(angle_deg, cue_xy)

            # 4) Acknowledge success and wait for 'wait' before sending coordinates
            send_message(sock, "100")
            wait_start = time.monotonic()
            wait_timeout = 10.0  # seconds to wait for 'wait' ack
            while True:
                ack = None
                try:
                    ack = receive_message(sock)
                except Exception:
                    ack = None
                if ack:
                    if ack.strip().lower() == "wait":
                        break
                    # store unexpected ack for debugging but keep waiting
                    reply = ack
                if wait_timeout is not None and (time.monotonic() - wait_start) > wait_timeout:
                    reply = reply or "timeout_waiting_WAIT"
                    error_msg = "等待機械手臂 wait 訊息逾時"
                    return balls_data, result, payload, reply, error_msg

            send_message(sock, payload)

            # 5) Wait for DONE
            start = time.monotonic()
            while True:
                try:
                    msg = receive_message(sock)
                except Exception:
                    msg = None
                if msg:
                    reply = msg
                    if msg == "DONE":
                        break
                if wait_done_max is not None and (time.monotonic() - start) > wait_done_max:
                    reply = reply or "timeout_waiting_DONE"
                    if error_msg is None:
                        error_msg = "等待機械手臂 DONE 訊息逾時"
                    break
        else:
            # Notify robot of failure with 200 and record error for logs
            payload = None
            reply = "NO_PLAN"
            try:
                send_message(sock, "200")
            except Exception:
                pass
            err_msg = get_last_plan_shot_error()
            error_msg = err_msg or "路徑規劃失敗，請稍後重試"
            if err_msg:
                print(f"[perform_handshaked_strike] 規劃失敗：{err_msg}")

    finally:
        try:
            if sock:
                sock.close()
        except Exception:
            pass

    return balls_data, result, payload, reply, error_msg


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def capture_and_plan(*, target="min", intrinsics_path: Optional[str] = None,
                     render_dir: Optional[Path] = None) -> Tuple[dict, Optional[Tuple[float, Tuple[float, float]]]]:
    # Perform capture
    json_path, data = capture_balls(
        wait_sec=3,
        show=False,
        intrinsics_path=intrinsics_path,
        preview=False,
        cam_index=getattr(Config, 'CAMERA_INDEX', 0),
        cam_width=getattr(Config, 'CAMERA_WIDTH', None),
        cam_height=getattr(Config, 'CAMERA_HEIGHT', None),
    )
    if not json_path or not data:
        return {"timestamp": datetime.utcnow().isoformat(), "balls": []}, None

    # Compute plan
    result = plan_shot_from_json(json_path, target, show=False)
    return data, result
