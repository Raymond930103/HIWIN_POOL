"""
Live camera YOLO test for tools/0918best.pt

Usage examples:
  python tools/live_cam_0918.py
  python tools/live_cam_0918.py --cam 1 --conf 0.35
  python tools/live_cam_0918.py --weights tools/0918best.pt

Keys:
  q: quit
  s: save current frame to tools/captures/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


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


def run(
    cam: int | str = 0,
    weights: str = "tools/0918best.pt",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    view_w: int | None = None,
    view_h: int | None = None,
    show_fps: bool = True,
    iou_thresh: float = 0.5,
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

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame; retrying...")
            time.sleep(0.02)
            continue

        h, w = frame.shape[:2]
        thick, font_scale = _auto_thickness(h, w)

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
                color = _color_for(cid)
                label_name = (
                    names.get(cid, cid) if isinstance(names, dict)
                    else (names[cid] if names and cid < len(names) else cid)
                )
                label = f"{label_name} {cf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=thick)
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
                y_text = max(y1, th + 3)
                cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 6, y_text + 2), color, -1)
                cv2.putText(frame, label, (x1 + 3, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick)

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
    ap.add_argument("--weights", default="tools/0918best.pt", help="Path to model weights")
    ap.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.50, help="IoU threshold for overlap suppression")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--device", default=None, help="Device, e.g. '0' for GPU or 'cpu'")
    ap.add_argument("--width", type=int, default=None, help="Capture/display width")
    ap.add_argument("--height", type=int, default=None, help="Capture/display height")
    ap.add_argument("--no-fps", action="store_true", help="Hide FPS overlay")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Allow numeric cam index
    cam_src: int | str
    try:
        cam_src = int(args.cam)
    except (TypeError, ValueError):
        cam_src = args.cam

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
    )
