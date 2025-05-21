#!/usr/bin/env python3
"""
auto_corners.py ───────────────────────────────────────────────────────────
根據單張相機影像，自動偵測「球可滾動範圍」四角座標並輸出 corner.json。

支援兩種方法：
1. hsv   ─ 利用桌布的綠色遮罩找到外框，再往內縮邊庫厚度 (inner‑cm)。
2. aruco ─ 若在四個 playfield 角貼了 ArUco ID=0,1,2,3，直接取標籤中心。

產生的 corner.json 可直接交給 calc_homography.py 進一步計算 H 矩陣。

使用範例
────────
$ python auto_corners.py frame.jpg --inner-cm 5        # hsv 法 (預設)
$ python auto_corners.py frame.jpg --method aruco      # ArUco 法
$ python auto_corners.py frame.jpg -o mycorner.json -d # 開啟除錯視窗

參數說明
────────
 image          拍到整張桌子的單張圖片（建議先畸變校正）。
 -o / --out     corner.json 的檔名 (預設 corner.json)。
 --method       hsv | aruco        (預設 hsv)。
 --inner-cm     邊庫厚度 (cm)。=0 表示取外框，不內縮。
 --debug (-d)   顯示偵測結果視窗。

作者：ChatGPT (o3)
"""
from __future__ import annotations
import cv2
import json
import argparse
import numpy as np
import sys
from pathlib import Path

# －－－－－－ 使用者可自行調整的預設值 －－－－－－
TABLE_W_CM = 73.5  # 桌面含邊庫外框寬 (cm)
TABLE_H_CM = 37.5  # 桌面含邊庫外框高 (cm)
ARUCO_DICT = cv2.aruco.DICT_4X4_50  # 若用 aruco，可改其他字典
# －－－－－－ ------------------------------------------------


def save_json(pts: np.ndarray, out_path: str = "corner.json") -> None:
    """將四角 (4,2) numpy 轉成 corner.json 格式。"""
    tl, tr, br, bl = pts.astype(float)
    data = {
        "top_left":     {"x": tl[0], "y": tl[1]},
        "top_right":    {"x": tr[0], "y": tr[1]},
        "bottom_right": {"x": br[0], "y": br[1]},
        "bottom_left":  {"x": bl[0], "y": bl[1]},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {out_path}")


# ──────────────────────────────────────────────────────────────
# 方法 A：桌布綠色遮罩 + 內縮邊庫
# ──────────────────────────────────────────────────────────────

def detect_by_hsv(img: np.ndarray, debug: bool = False) -> np.ndarray:
    """回傳外框四點 (tl, tr, br, bl)，shape=(4,2)。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 依桌布顏色調整，下方範圍涵蓋大部分綠墊
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # 形態學閉運算補洞，避免袋口／反光破洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 取最大連通輪廓 ≈ 桌面 + 邊庫
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("找不到綠色桌面，請調整 HSV 閾值")
    cnt = max(cnts, key=cv2.contourArea)

    # 若逼近多邊形剛好 4 點就用；否則取最小外接矩形
    peri = cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(poly) == 4:
        pts = poly.reshape(-1, 2)
    else:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect)

    # 排序 (tl, tr, br, bl)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    pts_ordered = np.array([tl, tr, br, bl], dtype=np.float32)

    if debug:
        dbg = img.copy()
        for i, p in enumerate(pts_ordered):
            cv2.circle(dbg, tuple(p.astype(int)), 10, (0, 0, 255), -1)
            cv2.putText(dbg, "TLTRBRBL"[i*2:i*2+2], tuple(p.astype(int) + [5, -5]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("outer", dbg)
        cv2.waitKey(0)
    return pts_ordered


def shrink_to_inner(pts_outer: np.ndarray, inner_cm: float) -> np.ndarray:
    """把外框四角往中心縮 inner_cm，回傳內框 4 點。"""
    if inner_cm <= 0:
        return pts_outer.astype(np.float32)

    # 中心點
    cx, cy = pts_outer.mean(axis=0)

    # 估算 px_per_cm（平均寬高）
    w_px = np.linalg.norm(pts_outer[1] - pts_outer[0])
    h_px = np.linalg.norm(pts_outer[2] - pts_outer[1])
    px_per_cm = (w_px / TABLE_W_CM + h_px / TABLE_H_CM) / 2
    delta = inner_cm * px_per_cm

    pts_inner = []
    for p in pts_outer:
        v = np.array([cx, cy]) - p
        v /= np.linalg.norm(v)  # 單位向量
        pts_inner.append(p + v * delta)
    return np.array(pts_inner, dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# 方法 B：ArUco 標籤
# ──────────────────────────────────────────────────────────────

def detect_by_aruco(img: np.ndarray, debug: bool = False) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(img)
    if ids is None or len(ids) < 4:
        raise RuntimeError("找不到足夠的 ArUco 標籤 (需要 ID=0,1,2,3)")

    centers = {int(i): c[0].mean(axis=0) for i, c in zip(ids, corners)}
    try:
        tl, tr, br, bl = (centers[i] for i in (0, 1, 2, 3))
    except KeyError:
        raise RuntimeError("請確保四角分別貼 ID=0,1,2,3 的標籤")

    pts = np.array([tl, tr, br, bl], dtype=np.float32)
    if debug:
        dbg = img.copy()
        for p in pts:
            cv2.circle(dbg, tuple(np.int32(p)), 12, (255, 0, 0), -1)
        cv2.imshow("aruco", dbg)
        cv2.waitKey(0)
    return pts


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("自動偵測桌角 → corner.json")
    ap.add_argument("image", help="拍到整張桌子的單張照片（建議先 undistort）")
    ap.add_argument("-o", "--out", default="corner.json", help="輸出檔名")
    ap.add_argument("--method", choices=["hsv", "aruco"], default="hsv",
                    help="hsv＝靠桌布綠色；aruco＝貼四張標籤")
    ap.add_argument("--inner-cm", type=float, default=5.0,
                    help="邊庫厚度 (cm)，0 表示不內縮 (含邊庫)")
    ap.add_argument("-d", "--debug", action="store_true", help="顯示偵測結果")
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        sys.exit(f"找不到影像檔 {args.image}")

    if args.method == "aruco":
        pts = detect_by_aruco(img, args.debug)
    else:
        outer = detect_by_hsv(img, args.debug)
        pts = shrink_to_inner(outer, args.inner_cm)
        if args.debug:
            dbg = img.copy()
            for p in pts:
                cv2.circle(dbg, tuple(np.int32(p)), 10, (0, 255, 255), -1)
            cv2.imshow("inner", dbg)
            cv2.waitKey(0)

    save_json(pts, args.out)


if __name__ == "__main__":
    main()
