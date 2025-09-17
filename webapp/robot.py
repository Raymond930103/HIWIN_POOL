import os
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Make project root importable
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.vision.yoloball import capture_balls
from main.run_shot import plan_shot_from_json
from main.configs.setting import HOST, PORT
from main.communicate.tcp import create_connection, send_message
from .config import Config


def compute_arm_payload(angle_deg: float, cue_xy_m: Tuple[float, float]) -> str:
    # Mirror logic from main/main.py (without j6_diff adj)
    arm_angle = -angle_deg
    arm_x = round(cue_xy_m[0] * 1000, 2)        # m → mm
    arm_y = round(375 - cue_xy_m[1] * 1000, 2)  # bottom-origin conversion
    return f"{arm_angle:.2f}, {arm_x:.2f}, {arm_y:.2f}"


def send_to_robot(payload: str) -> bool:
    try:
        sock = create_connection(HOST, PORT)
        if sock is None:
            return False
        send_message(sock, payload)
        return True
    except Exception:
        return False


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
