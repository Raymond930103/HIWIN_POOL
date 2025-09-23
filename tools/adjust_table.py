#!/usr/bin/env python3
"""
Adjust Table Corners & Pockets
──────────────────────────────
- Interactive tool to tweak the 4 table corners and 6 pocket centers.
- Saves directly to the files the main code uses by default:
  - corners → `main/vision/corner.json`
  - pockets → `main/vision/pockets.json` (optional; for future use)

Usage
-----
1) With a still image:
   $ python tools/adjust_table.py --image path/to/frame.jpg

2) With live camera snapshot (default cam index 0):
   $ python tools/adjust_table.py

Keys
----
- m: toggle edit mode (corners ↔ pockets)
- drag with left mouse: move selected point
- s or Space: save JSONs
- r: reset pockets from current corners (recompute 4 corners + 2 midpoints)
- c: reset corners to a small centered rectangle
- q or Esc: quit

Notes
-----
- If `main/vision/corner.json` exists, it is loaded as initial corners; otherwise
  corners are initialized to image margins.
- If `main/vision/pockets.json` exists, it is loaded; otherwise pockets are
  derived from current corners as 4 corners + top/bottom midpoints.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


DEFAULT_CORNER_JSON = Path("main/vision/corner.json")
DEFAULT_POCKETS_JSON = Path("main/vision/pockets.json")


def load_image(image_path: str | None, cam_index: int = 0) -> np.ndarray:
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise SystemExit(f"Image not found: {image_path}")
        return img
    # Try single-shot from camera
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        # fallback to a blank canvas
        print("[WARN] Camera not available; using blank canvas 1280x720")
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[WARN] Failed to grab frame; using blank canvas 1280x720")
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    return frame


def load_corners(json_path: Path, img_shape: Tuple[int, int]) -> np.ndarray:
    h, w = img_shape[:2]
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            c = json.load(f)
        pts = np.array([
            [c["top_left"]["x"], c["top_left"]["y"]],
            [c["top_right"]["x"], c["top_right"]["y"]],
            [c["bottom_right"]["x"], c["bottom_right"]["y"]],
            [c["bottom_left"]["x"], c["bottom_left"]["y"]],
        ], dtype=float)
        return pts
    # default: inset rectangle near image borders (50 px margin)
    m = 50
    return np.array([[m, m], [w - m, m], [w - m, h - m], [m, h - m]], dtype=float)


def corners_to_json(pts: np.ndarray) -> dict:
    return {
        "top_left": {"x": float(pts[0, 0]), "y": float(pts[0, 1])},
        "top_right": {"x": float(pts[1, 0]), "y": float(pts[1, 1])},
        "bottom_right": {"x": float(pts[2, 0]), "y": float(pts[2, 1])},
        "bottom_left": {"x": float(pts[3, 0]), "y": float(pts[3, 1])},
    }


def load_pockets(json_path: Path, corners: np.ndarray) -> np.ndarray:
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # support either dict by names or list of 6 points
        if isinstance(d, dict):
            names = [
                "top_left",
                "top_right",
                "bottom_right",
                "bottom_left",
                "top_mid",
                "bottom_mid",
            ]
            pts = []
            for k in names:
                v = d.get(k)
                if not v or "x" not in v or "y" not in v:
                    raise ValueError(f"Invalid pockets.json entry for {k}")
                pts.append([float(v["x"]), float(v["y"])])
            return np.array(pts, dtype=float)
        elif isinstance(d, list) and len(d) == 6:
            return np.array([[float(p[0]), float(p[1])] for p in d], dtype=float)
    # default from corners: 4 corners + top/bottom midpoints
    tl, tr, br, bl = corners
    top_mid = (tl + tr) / 2.0
    bot_mid = (bl + br) / 2.0
    return np.array([tl, tr, br, bl, top_mid, bot_mid], dtype=float)


def pockets_to_json(pts: np.ndarray) -> dict:
    names = [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
        "top_mid",
        "bottom_mid",
    ]
    data = {}
    for name, (x, y) in zip(names, pts):
        data[name] = {"x": float(x), "y": float(y)}
    return data


def clamp_point(p: np.ndarray, w: int, h: int) -> np.ndarray:
    x = float(np.clip(p[0], 0, w - 1))
    y = float(np.clip(p[1], 0, h - 1))
    return np.array([x, y], dtype=float)


def main():
    ap = argparse.ArgumentParser("Adjust table corners and pockets")
    ap.add_argument("--image", "-i", help="Background image file; if omitted, grabs one frame from camera")
    ap.add_argument("--cam", type=int, default=0, help="Camera index when --image not provided (default: 0)")
    ap.add_argument("--corner-json", default=str(DEFAULT_CORNER_JSON), help="Path to corner.json to read/write")
    ap.add_argument("--pockets-json", default=str(DEFAULT_POCKETS_JSON), help="Path to pockets.json to read/write")
    ap.add_argument("--pocket-radius", type=int, default=50, help="Pocket radius for preview (pixels)")
    ap.add_argument("--corner-reset-scale", type=float, default=0.5,
                    help="Size of centered rectangle as a fraction of image width/height (default: 0.5)")
    args = ap.parse_args()

    img = load_image(args.image, args.cam)
    H, W = img.shape[:2]

    corner_path = Path(args.corner_json)
    pockets_path = Path(args.pockets_json)

    corners = load_corners(corner_path, img.shape)
    pockets = load_pockets(pockets_path, corners)

    # Keep originals for optional reset of corners
    corners_orig = corners.copy()

    mode = "corners"  # or "pockets"
    selected_idx: int | None = None
    hit_radius = 16

    def draw() -> np.ndarray:
        vis = img.copy()
        # Draw corners polygon and points
        poly = corners.astype(int).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], True, (0, 255, 0), 2)
        for i, p in enumerate(corners):
            color = (0, 0, 255) if mode == "corners" else (80, 80, 80)
            cv2.circle(vis, tuple(p.astype(int)), 5, color, -1)
            cv2.putText(vis, f"C{i}", tuple(p.astype(int) + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # Draw pockets
        for i, p in enumerate(pockets):
            color = (255, 0, 255) if mode == "pockets" else (120, 120, 120)
            cv2.circle(vis, tuple(p.astype(int)), args.pocket_radius, color, 2)
            cv2.putText(vis, f"P{i}", tuple(p.astype(int) + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # HUD text
        hud = [
            f"Mode: {mode} | m: toggle | drag: move | s/Space: save | r: reset pockets | c: center corners | q/Esc: quit",
            f"corner.json → {corner_path}",
            f"pockets.json → {pockets_path}",
        ]
        y = 24
        for line in hud:
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 220, 30), 2)
            y += 22
        return vis

    def mouse_cb(event, x, y, flags, _userdata):
        nonlocal selected_idx, corners, pockets
        if event == cv2.EVENT_LBUTTONDOWN:
            pts = corners if mode == "corners" else pockets
            dists = np.linalg.norm(pts - np.array([x, y], dtype=float), axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] < hit_radius:
                selected_idx = idx
        elif event == cv2.EVENT_MOUSEMOVE and selected_idx is not None:
            if mode == "corners":
                corners[selected_idx] = clamp_point(np.array([x, y], dtype=float), W, H)
            else:
                pockets[selected_idx] = clamp_point(np.array([x, y], dtype=float), W, H)
        elif event == cv2.EVENT_LBUTTONUP:
            selected_idx = None

    win = "Adjust Table"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_cb)
    print("[INFO] Drag points. 'm' toggle mode, 's'/Space to save, 'r' reset pockets, 'q'/Esc to quit.")

    while True:
        cv2.imshow(win, draw())
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        elif key in (ord("m"), ord("M")):
            mode = "pockets" if mode == "corners" else "corners"
        elif key in (ord("s"), ord("S"), 32):  # save
            # Save corners
            corner_path.parent.mkdir(parents=True, exist_ok=True)
            with open(corner_path, "w", encoding="utf-8") as f:
                json.dump(corners_to_json(corners), f, ensure_ascii=False, indent=2)
            # Save pockets
            pockets_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pockets_path, "w", encoding="utf-8") as f:
                json.dump(pockets_to_json(pockets), f, ensure_ascii=False, indent=2)
            print(f"[Saved] {corner_path} and {pockets_path}")
            vis = draw()
            cv2.putText(vis, "Saved!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)
            cv2.imshow(win, vis)
            cv2.waitKey(800)
        elif key in (ord("r"), ord("R")):
            # recompute pockets from current corners
            tl, tr, br, bl = corners
            top_mid = (tl + tr) / 2.0
            bot_mid = (bl + br) / 2.0
            pockets = np.array([tl, tr, br, bl, top_mid, bot_mid], dtype=float)
        elif key in (ord("c"), ord("C")):
            # reset corners to a centered small rectangle
            scale = float(np.clip(args.corner_reset_scale, 0.1, 0.9))
            w_box = W * scale
            h_box = H * scale
            cx, cy = W / 2.0, H / 2.0
            x1 = cx - w_box / 2.0
            x2 = cx + w_box / 2.0
            y1 = cy - h_box / 2.0
            y2 = cy + h_box / 2.0
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
            # also refresh pockets from these new corners
            tl, tr, br, bl = corners
            top_mid = (tl + tr) / 2.0
            bot_mid = (bl + br) / 2.0
            pockets = np.array([tl, tr, br, bl, top_mid, bot_mid], dtype=float)
        elif key in (ord("b"), ord("B")):
            # optional: revert corners to original
            corners = corners_orig.copy()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
