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
   $ python tools/adjust_table.py --image path/to/frame.jpg [--intrinsics main/vision/intrinsics.yaml]

2) With live camera (continuous):
   $ python tools/adjust_table.py --live --cam 0 [--width 1920 --height 1080] \
                                  [--intrinsics main/vision/intrinsics.yaml]

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

Undistort
---------
- Provide `--intrinsics` to undistort the preview (still or live) using `intrinsics.yaml`.
- Edited points are stored in the original (distorted) pixel space to remain
  compatible with the main pipeline. The tool converts between spaces internally
  so what you drag on the undistorted view is saved correctly for the detector.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import yaml


DEFAULT_CORNER_JSON = Path("main/vision/corner.json")
DEFAULT_POCKETS_JSON = Path("main/vision/pockets.json")


# ──────────────────────────────────────────────────────────────
# Intrinsics helpers (optional undistort for live/preview)
# ──────────────────────────────────────────────────────────────
class UndistortCfg:
    def __init__(self, K: np.ndarray, D: np.ndarray, newK: np.ndarray,
                 map1: Optional[np.ndarray], map2: Optional[np.ndarray], size: Tuple[int, int]):
        self.K = K
        self.D = D
        self.newK = newK
        self.map1 = map1
        self.map2 = map2
        self.size = size  # (w, h)


def load_intrinsics_yaml(p: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(p, 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f)
    # Accept either {K: [...], D: [...]} or nested dicts from OpenCV
    M = d.get('camera_matrix', d.get('K'))
    if isinstance(M, dict):
        M = M.get('data', M.get('vals', None))
    K = np.array(M, dtype=np.float32).reshape(3, 3)
    D_ = d.get('distortion_coefficients', d.get('dist_coeff', d.get('D')))
    if isinstance(D_, dict):
        D_ = D_.get('data', D_.get('vals', None))
    D = np.array(D_, dtype=np.float32).reshape(-1)
    return K, D


def make_undistort_cfg(K: np.ndarray, D: np.ndarray, w: int, h: int, alpha: float = 0.0) -> UndistortCfg:
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), float(alpha))
    # Maps from undistorted (dst) → distorted (src)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_32FC1)
    return UndistortCfg(K, D, newK, map1, map2, (w, h))


def undistort_image(img: np.ndarray, cfg: Optional[UndistortCfg]) -> np.ndarray:
    if cfg is None:
        return img
    # Use remap for consistency with map direction we hold
    return cv2.remap(img, cfg.map1, cfg.map2, interpolation=cv2.INTER_LINEAR)


def pts_distorted_to_undistorted(pts: np.ndarray, cfg: Optional[UndistortCfg]) -> np.ndarray:
    """Convert points defined in original distorted pixel space → undistorted pixel space.
    If cfg is None, returns input.
    """
    if cfg is None or pts.size == 0:
        return pts.copy()
    arr = pts.reshape(-1, 1, 2).astype(np.float32)
    ud = cv2.undistortPoints(arr, cfg.K, cfg.D, P=cfg.newK)
    return ud.reshape(-1, 2).astype(np.float32)


def pt_undistorted_to_distorted(u: float, v: float, cfg: Optional[UndistortCfg]) -> Tuple[float, float]:
    """Map a pixel from undistorted view back to original distorted pixel coords using map1/map2.
    If cfg is None, returns (u, v).
    """
    if cfg is None:
        return float(u), float(v)
    w, h = cfg.size
    x = int(np.clip(round(u), 0, w - 1))
    y = int(np.clip(round(v), 0, h - 1))
    # map1/map2 hold src-x/src-y for each undistorted dst pixel
    src_x = float(cfg.map1[y, x])
    src_y = float(cfg.map2[y, x])
    return src_x, src_y


def load_image(image_path: str | None, cam_index: int = 0, *, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise SystemExit(f"Image not found: {image_path}")
        return img
    # Try single-shot from camera
    cap = cv2.VideoCapture(cam_index)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
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
    ap.add_argument("--live", action="store_true", help="Show live camera feed instead of a single snapshot")
    ap.add_argument("--width", type=int, default=None, help="Camera capture width (live/snapshot)")
    ap.add_argument("--height", type=int, default=None, help="Camera capture height (live/snapshot)")
    ap.add_argument("--intrinsics", default=None, help="Path to intrinsics.yaml to undistort preview (optional)")
    ap.add_argument("--alpha", type=float, default=0.0, help="Undistort alpha [0..1], 0 keeps FOV tight")
    ap.add_argument("--corner-json", default=str(DEFAULT_CORNER_JSON), help="Path to corner.json to read/write")
    ap.add_argument("--pockets-json", default=str(DEFAULT_POCKETS_JSON), help="Path to pockets.json to read/write")
    ap.add_argument("--pocket-radius", type=int, default=50, help="Pocket radius for preview (pixels)")
    ap.add_argument("--corner-reset-scale", type=float, default=0.5,
                    help="Size of centered rectangle as a fraction of image width/height (default: 0.5)")
    args = ap.parse_args()

    # Acquire initial frame and prepare optional undistortion
    img = load_image(args.image, args.cam, width=args.width, height=args.height)
    H, W = img.shape[:2]
    und_cfg: Optional[UndistortCfg] = None
    # Auto-pick default intrinsics if not provided
    intr_path = args.intrinsics
    if intr_path is None and Path("main/vision/intrinsics.yaml").exists():
        intr_path = "main/vision/intrinsics.yaml"
    if intr_path:
        try:
            K, D = load_intrinsics_yaml(intr_path)
            und_cfg = make_undistort_cfg(K, D, W, H, alpha=float(args.alpha))
            print(f"[Undistort] Using {intr_path}")
        except Exception as e:
            print(f"[WARN] Failed to load intrinsics from {intr_path}: {e}")
            und_cfg = None

    corner_path = Path(args.corner_json)
    pockets_path = Path(args.pockets_json)

    corners = load_corners(corner_path, img.shape)
    pockets = load_pockets(pockets_path, corners)

    # Keep originals for optional reset of corners
    corners_orig = corners.copy()

    mode = "corners"  # or "pockets"
    selected_idx: int | None = None
    hit_radius = 16

    def draw(frame_bgr: np.ndarray) -> np.ndarray:
        # Apply undistortion to background if configured
        vis = undistort_image(frame_bgr, und_cfg)
        # Convert stored (distorted) points to display (undistorted) space for drawing
        draw_corners = pts_distorted_to_undistorted(corners, und_cfg).astype(int)
        draw_pockets = pts_distorted_to_undistorted(pockets, und_cfg).astype(int)
        # Draw corners polygon and points
        poly = draw_corners.reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], True, (0, 255, 0), 2)
        for i, p in enumerate(draw_corners):
            color = (0, 0, 255) if mode == "corners" else (80, 80, 80)
            cv2.circle(vis, tuple(p.astype(int)), 5, color, -1)
            cv2.putText(vis, f"C{i}", tuple(p.astype(int) + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # Draw pockets
        for i, p in enumerate(draw_pockets):
            color = (255, 0, 255) if mode == "pockets" else (120, 120, 120)
            cv2.circle(vis, tuple(p.astype(int)), args.pocket_radius, color, 2)
            cv2.putText(vis, f"P{i}", tuple(p.astype(int) + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # HUD text
        hud = [
            f"Mode: {mode} | m: toggle | drag: move | s/Space: save | r: reset pockets | c: center corners | q/Esc: quit",
            f"corner.json → {corner_path}",
            f"pockets.json → {pockets_path}",
            (f"Undistort: ON ({args.intrinsics})" if und_cfg else "Undistort: OFF"),
        ]
        y = 24
        for line in hud:
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 220, 30), 2)
            y += 22
        return vis

    def mouse_cb(event, x, y, flags, _userdata):
        nonlocal selected_idx, corners, pockets
        if event == cv2.EVENT_LBUTTONDOWN:
            # hit-test in display (undistorted) space
            disp_pts = pts_distorted_to_undistorted(corners if mode == "corners" else pockets, und_cfg)
            dists = np.linalg.norm(disp_pts - np.array([x, y], dtype=float), axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] < hit_radius:
                selected_idx = idx
        elif event == cv2.EVENT_MOUSEMOVE and selected_idx is not None:
            # Convert mouse position (undistorted) back to distorted for storage
            dx, dy = pt_undistorted_to_distorted(float(x), float(y), und_cfg)
            if mode == "corners":
                corners[selected_idx] = clamp_point(np.array([dx, dy], dtype=float), W, H)
            else:
                pockets[selected_idx] = clamp_point(np.array([dx, dy], dtype=float), W, H)
        elif event == cv2.EVENT_LBUTTONUP:
            selected_idx = None

    win = "Adjust Table"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)
    print("[INFO] Drag points. 'm' toggle mode, 's'/Space to save, 'r' reset pockets, 'q'/Esc to quit.")

    # Prepare camera for live mode
    cap = None
    if args.live and not args.image:
        cap = cv2.VideoCapture(args.cam)
        if args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
        if args.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
        if not cap.isOpened():
            print("[WARN] Live camera not available; falling back to single snapshot mode")
            cap = None

    while True:
        frame = img
        if cap is not None:
            ok, frm = cap.read()
            if ok:
                frame = frm
            else:
                print("[WARN] Failed to read live frame; showing last frame")

        cv2.imshow(win, draw(frame))
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

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
