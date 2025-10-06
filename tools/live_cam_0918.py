"""
Live camera YOLO test for tools/0918best.pt

Usage examples:
  python tools/live_cam_0918.py
  python tools/live_cam_0918.py --cam 1 --conf 0.35
  python tools/live_cam_0918.py --weights tools/0918best.pt
  python tools/live_cam_0918.py --intrinsics main/vision/intrinsics.yaml

Keys:
  q: quit
  s: save current frame to tools/captures/
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import yaml


# ===================== Manual Mapping & Colors =====================
# Edit this mapping to tell the script which detected class corresponds
# to which real pool ball number (1–15). Keys can be either the class id
# (int) or the class name (str) from your YOLO model.
#
# Examples (uncomment/adapt to your model):
# CLASS_TO_BALL = {
#     0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8,
#     8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15,
#     # or by names if your model uses strings like 'ball_1', 'stripe_9', etc.
#     '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
#     '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,
#     'ball_1': 1, 'ball_2': 2, 'ball_3': 3, 'ball_4': 4, 'ball_5': 5,
#     'ball_6': 6, 'ball_7': 7, 'ball_8': 8, 'ball_9': 9, 'ball_10': 10,
#     'ball_11': 11, 'ball_12': 12, 'ball_13': 13, 'ball_14': 14, 'ball_15': 15,
# }
CLASS_TO_BALL: dict[int | str, int] = {}

# Standard pool ball colors in BGR (OpenCV). Adjust if you prefer.
# 1: yellow, 2: blue, 3: red, 4: purple, 5: orange, 6: green,
# 7: maroon/burgundy, 8: black, 9–15: same hues as 1–7 (stripes).
BALL_NUMBER_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    1: (0, 220, 255),   # yellow
    2: (255, 120, 0),   # blue
    3: (0, 0, 220),     # red
    4: (180, 60, 180),  # purple
    5: (0, 140, 255),   # orange
    6: (0, 150, 0),     # green
    7: (30, 30, 120),   # maroon/burgundy (dark reddish)
    8: (0, 0, 0),       # black
    9: (0, 220, 255),   # yellow (stripe)
    10: (255, 120, 0),  # blue (stripe)
    11: (0, 0, 220),    # red (stripe)
    12: (180, 60, 180), # purple (stripe)
    13: (0, 140, 255),  # orange (stripe)
    14: (0, 150, 0),    # green (stripe)
    15: (30, 30, 120),  # maroon (stripe)
}


def _auto_thickness(h: int, w: int) -> tuple[int, float]:
    base = int(round(max(h, w) / 300))
    base = max(1, min(base, 5))
    return base, max(0.4, min(0.8, base * 0.15))


def _color_for(cls_id: int) -> tuple[int, int, int]:
    # Simple deterministic color palette
    palette = (
        (255, 56, 56), (255, 159, 56), (255, 255, 56), (56, 255, 56),
        (56, 255, 255), (56, 56, 255), (255, 56, 255), (180, 130, 70),
        (80, 175, 76), (92, 125, 179), (164, 73, 163), (230, 230, 230)
    )
    return palette[cls_id % len(palette)]


def _ball_num_from_class(
    cls_id: int,
    names: dict[int, str] | list[str] | None,
) -> int | None:
    # 1) explicit numeric id mapping
    if cls_id in CLASS_TO_BALL:
        return CLASS_TO_BALL[cls_id]

    # 2) try name-based mapping
    label_name: str | None = None
    if isinstance(names, dict):
        label_name = names.get(cls_id)
    elif isinstance(names, list) and 0 <= cls_id < len(names):
        label_name = names[cls_id]

    if label_name:
        # direct name key
        if label_name in CLASS_TO_BALL:
            return CLASS_TO_BALL[label_name]
        # case-insensitive variants
        low = label_name.lower()
        up = label_name.upper()
        if low in CLASS_TO_BALL:
            return CLASS_TO_BALL[low]
        if up in CLASS_TO_BALL:
            return CLASS_TO_BALL[up]
        # 3) heuristic: parse first integer in name (e.g., 'ball_12' -> 12)
        m = re.search(r"\d+", label_name)
        if m:
            try:
                n = int(m.group(0))
                if 1 <= n <= 15:
                    return n
            except ValueError:
                pass
    return None


def _color_for_ball_num(ball_num: int) -> tuple[int, int, int]:
    return BALL_NUMBER_COLORS_BGR.get(ball_num, _color_for(ball_num))


def _text_color_for_bg(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    # Choose black or white text based on luminance for contrast
    b, g, r = bgr
    # Perceptual luminance (approx)
    y = 0.114 * b + 0.587 * g + 0.299 * r
    return (255, 255, 255) if y < 128 else (0, 0, 0)


def _load_intrinsics(p: str):
    with open(p, 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f)
    Kd = d.get('camera_matrix', d.get('K'))
    if isinstance(Kd, dict):
        Kd = Kd.get('data', Kd.get('vals', None))
    K = np.array(Kd, dtype=np.float32).reshape(3, 3)
    Dd = d.get('distortion_coefficients', d.get('dist_coeff', d.get('D')))
    if isinstance(Dd, dict):
        Dd = Dd.get('data', Dd.get('vals', None))
    D = np.array(Dd, dtype=np.float32).reshape(-1)
    return K, D


def run(
    cam: int | str = 0,
    weights: str = "tools/best.pt",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    view_w: int | None = None,
    view_h: int | None = None,
    show_fps: bool = True,
    iou_thresh: float = 0.5,
    intrinsics_path: str | None = None,
    alpha: float = 0.0,
):
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    names = model.names if hasattr(model, "names") else None

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {cam}")

    if view_w:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(view_w))
    if view_h:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(view_h))

    out_dir = Path("tools/captures")
    out_dir.mkdir(parents=True, exist_ok=True)

    win = "YOLO Live (0918best)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    t_last = time.time()
    fps = 0.0
    map1 = map2 = None
    use_undistort = False
    K = D = newK = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame; retrying...")
            time.sleep(0.02)
            continue

        h, w = frame.shape[:2]
        thick, font_scale = _auto_thickness(h, w)

        # Initialize undistort maps on first frame if intrinsics provided
        if intrinsics_path and not use_undistort:
            try:
                K, D = _load_intrinsics(intrinsics_path)
                newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), float(alpha))
                map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_16SC2)
                use_undistort = True
                print(f"[Undistort] Using intrinsics from {intrinsics_path}")
            except Exception as e:
                print(f"[WARN] Failed to load intrinsics: {e}")

        if use_undistort and map1 is not None and map2 is not None:
            frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        # Run inference
        res = model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            agnostic_nms=True,
            device=device,
            verbose=False,
        )[0]

        # Draw detections (keep only highest-confidence for overlapping boxes)
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)

            # Greedy class-agnostic NMS: keep highest conf when boxes overlap (same place)
            order = np.argsort(-confs)  # high -> low
            keep = []
            def iou(a, b):
                ix1 = max(a[0], b[0])
                iy1 = max(a[1], b[1])
                ix2 = min(a[2], b[2])
                iy2 = min(a[3], b[3])
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                if inter <= 0:
                    return 0.0
                area_a = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
                area_b = max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))
                union = area_a + area_b - inter
                return inter / union if union > 0 else 0.0

            for idx in order:
                a = xyxy[idx]
                if all(iou(a, xyxy[j]) <= iou_thresh for j in keep):
                    keep.append(idx)

            for i in keep:
                x1, y1, x2, y2 = xyxy[i]
                cf = float(confs[i])
                cid = int(clss[i])
                # Prefer ball-number-based color if we can map it
                ball_num = _ball_num_from_class(cid, names)
                color = _color_for_ball_num(ball_num) if ball_num is not None else _color_for(cid)
                label_name = (
                    names.get(cid, cid) if isinstance(names, dict)
                    else (names[cid] if names and cid < len(names) else cid)
                )
                label_main = f"{label_name}"
                if ball_num is not None:
                    label_main += f"#{ball_num}"
                label = f"{label_main} {cf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=thick)
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
                y_text = max(y1, th + 3)
                txt_col = _text_color_for_bg(color)
                cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 6, y_text + 2), color, -1)
                cv2.putText(frame, label, (x1 + 3, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_col, thick)

        # FPS overlay
        if show_fps:
            now = time.time()
            dt = now - t_last
            t_last = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
            txt = f"FPS: {fps:.1f}  conf>={conf:.2f}"
            cv2.putText(frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"frame_{ts}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"[Saved] {out_path}")

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser(description="Live camera detection with tools/0918best.pt")
    ap.add_argument("--cam", default=0, help="Camera index or stream URL (default: 0)")
    ap.add_argument("--weights", default="tools/best.pt", help="Path to model weights")
    ap.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.50, help="IoU threshold for overlap suppression")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--device", default=None, help="Device, e.g. '0' for GPU or 'cpu'")
    ap.add_argument("--width", type=int, default=None, help="Capture/display width")
    ap.add_argument("--height", type=int, default=None, help="Capture/display height")
    ap.add_argument("--no-fps", action="store_true", help="Hide FPS overlay")
    ap.add_argument("--intrinsics", default=None, help="Path to intrinsics.yaml for undistortion")
    ap.add_argument("--alpha", type=float, default=0.0, help="Undistort alpha [0..1]")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Allow numeric cam index
    cam_src: int | str
    try:
        cam_src = int(args.cam)
    except (TypeError, ValueError):
        cam_src = args.cam

    # Auto-pick intrinsics if not provided and default exists
    default_intr = Path("main/vision/intrinsics.yaml")
    intr = args.intrinsics or (str(default_intr) if default_intr.exists() else None)

    run(
        cam=cam_src,
        weights=args.weights,
        conf=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        view_w=args.width,
        view_h=args.height,
        show_fps=not args.no_fps,
        intrinsics_path=intr,
        alpha=args.alpha,
    )
