#!/usr/bin/env python3
"""
Simple frame grabber to collect calibration images from a live camera.

Usage:
  python tools/grab_calib.py --cam 0 --outdir tools/calib_imgs --width 1920 --height 1080

Keys:
  s / space: save current frame
  q / esc   : quit
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2


def run(cam=0, outdir="tools/calib_imgs", width=None, height=None):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(cam)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {cam}")

    win = "Calib Grabber"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Press s/space to save; q/esc to quit.")
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame; retrying…")
            time.sleep(0.02)
            continue

        h, w = frame.shape[:2]
        cv2.putText(frame, "s/space: save   q/esc: quit", (10, max(24, h - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2)
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('s'), ord(' ')):
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = out / f"calib_{ts}_{idx:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            idx += 1
            print(f"[Saved] {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Grab frames for calibration")
    ap.add_argument("--cam", default=0, help="Camera index or URL")
    ap.add_argument("--outdir", default="tools/calib_imgs", help="Output directory")
    ap.add_argument("--width", type=int, default=None, help="Capture width")
    ap.add_argument("--height", type=int, default=None, help="Capture height")
    args = ap.parse_args()

    try:
        cam_src = int(args.cam)
    except (TypeError, ValueError):
        cam_src = args.cam

    run(cam=cam_src, outdir=args.outdir, width=args.width, height=args.height)

