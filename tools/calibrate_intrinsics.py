#!/usr/bin/env python3
"""
Calibrate camera intrinsics (K, D) from a set of chessboard images and
save to YAML compatible with this repo (keys: K, D, reproj_error_px).

Usage examples:
  # 1) Capture a dataset of chessboard images first (see grab_calib.py)
  # 2) Run calibration:
  python tools/calibrate_intrinsics.py tools/calib_imgs \
      --pattern-cols 9 --pattern-rows 6 --square-mm 25 \
      --out main/vision/intrinsics.yaml --preview

Notes:
- pattern-cols/rows are the INNER corner counts (OpenCV convention).
- square-mm is the real side length of a chessboard square in millimeters.
- Provide a directory containing images, or a glob pattern to images.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml


def collect_image_paths(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in exts:
                paths.extend(sorted(p.glob(f"*{ext}")))
        else:
            # treat as glob
            for m in glob.glob(inp):
                mp = Path(m)
                if mp.suffix.lower() in exts:
                    paths.append(mp)
    # unique & sort
    paths = sorted({p.resolve() for p in paths})
    return paths


def calibrate_from_images(
    images: List[Path],
    pattern_size: Tuple[int, int],  # (cols, rows) inner corners
    square_mm: float,
    show_detect: bool = False,
):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= (square_mm / 1000.0)  # meters (units do not matter for intrinsics scale)

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    imsize = None

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )

    used = 0
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] Skipping unreadable image: {p}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if imsize is None:
            imsize = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if not ret:
            print(f"[INFO] No chessboard found: {p}")
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        used += 1

        if show_detect:
            disp = img.copy()
            cv2.drawChessboardCorners(disp, (cols, rows), corners2, True)
            cv2.imshow("detect", disp)
            cv2.waitKey(50)

    if show_detect:
        cv2.destroyAllWindows()

    if not objpoints or not imgpoints or imsize is None:
        raise RuntimeError("No valid chessboard detections. Check images and pattern size.")

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, imsize, None, None
    )

    # Compute per-view reprojection error
    per_view_errs: List[float] = []
    for i in range(len(objpoints)):
        imgpts2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(imgpoints[i], imgpts2, cv2.NORM_L2) / len(imgpts2)
        per_view_errs.append(float(err))

    return rms, K, D, per_view_errs, imsize


def save_yaml(out_path: Path, K: np.ndarray, D: np.ndarray, rms: float):
    data = {
        "K": [float(x) for x in K.reshape(-1)],
        "D": [float(x) for x in D.reshape(-1)],
        "reproj_error_px": float(rms),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"[Saved] {out_path}")


def undistort_preview(K: np.ndarray, D: np.ndarray, sample: Path):
    img = cv2.imread(str(sample))
    if img is None:
        print(f"[WARN] Cannot read preview image: {sample}")
        return
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1)
    und = cv2.undistort(img, K, D, None, newK)
    side = np.hstack([img, und])
    cv2.imshow("original | undistorted", side)
    print("Press any key to close preview window…")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser("Calibrate camera intrinsics from chessboard images")
    ap.add_argument("inputs", nargs="+", help="Directory and/or glob(s) to images")
    ap.add_argument("--pattern-cols", type=int, default=6, help="Inner corners horizontally (columns)")
    ap.add_argument("--pattern-rows", type=int, default=9, help="Inner corners vertically (rows)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="Chessboard square size in millimeters")
    ap.add_argument("--out", default="main/vision/intrinsics.yaml", help="Output YAML path")
    ap.add_argument("--show-detect", action="store_true", help="Show corner detections while processing")
    ap.add_argument("--preview", action="store_true", help="Show undistort preview from first good image")
    return ap.parse_args()


def main():
    args = parse_args()
    imgs = collect_image_paths(args.inputs)
    if not imgs:
        raise SystemExit("No input images found.")

    print(f"[INFO] Found {len(imgs)} images. Detecting {args.pattern_cols}x{args.pattern_rows} chessboard…")
    rms, K, D, per_view_errs, imsize = calibrate_from_images(
        imgs, (args.pattern_cols, args.pattern_rows), args.square_mm, show_detect=args.show_detect
    )
    print(f"[OK] RMS reprojection error: {rms:.4f} px; image size: {imsize}")
    if per_view_errs:
        print(f"[INFO] Per-view mean error (px): min {min(per_view_errs):.3f}, max {max(per_view_errs):.3f}, avg {sum(per_view_errs)/len(per_view_errs):.3f}")

    out_path = Path(args.out)
    save_yaml(out_path, K, D, rms)

    if args.preview:
        # find first successfully used image (heuristic: try all)
        undistort_preview(K, D, imgs[0])


if __name__ == "__main__":
    main()

