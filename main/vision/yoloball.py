#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 撞球偵測（Base = 球桌左上角）
───────────────────────────────────────────────────
‣ 使用 corner.json → Homography H(pixel→cm)，Base 原點 = 左上角 (0,0)。
‣ 偵測球心後直接輸出 Base‑XY (cm)。
‣ 口袋座標：若存在 pockets.json/pocket.json，優先採用其像素座標；
  否則以 H⁻¹(cm→pixel) 反推四角，再取 4 角 + 上下中點。
"""
from __future__ import annotations

import cv2, json, time, yaml, numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from ultralytics import YOLO

# === 參數 ===
CAM_URL     = 0
SAVE_DIR    = Path("captures_json"); SAVE_DIR.mkdir(exist_ok=True)
CONF_THRES  = 0.10
IOU_THRES   = 0.50
POCKET_R_PX = 50
TABLE_W_CM  = 73.5
TABLE_H_CM  = 37.5
MIN_SEP_CM  = 1.0
# Prefer the newer weights if present; fallback to legacy
MODEL_PATH  = "tools/0918best.pt" if Path("tools/0918best.pt").exists() else "main/vision/best2.pt"
CLASS_NAMES = ['2','2','2','3','3','14','6','3','3','2','4','3','3','0','1','1']
CORNER_JSON  = "main/vision/corner.json"
POCKETS_JSON = "main/vision/pockets.json"   # optional; falls back to computed
POCKET_JSON  = "main/vision/pocket.json"    # legacy singular filename support

# ═════════ 公開 API ═════════

def capture_balls(*,
                  wait_sec: int = 3,
                  show: bool = False,
                  intrinsics_path: str | None = None,
                  preview: bool = True,
                  cam_index: int | str = CAM_URL,
                  cam_width: Optional[int] = None,
                  cam_height: Optional[int] = None,
                 ) -> Tuple[str | None, dict | None]:
    """拍照→偵測→座標轉換→JSON；Esc 取消回 (None,None)

    preview: 是否顯示 OpenCV 倒數視窗（Web 伺服器請設 False）
    """

    # 1) 拍照
    K = D = None
    if intrinsics_path:
        K, D = _load_intrinsics(intrinsics_path)

    img = _snap(wait_sec, preview, cam_index=cam_index, cam_width=cam_width, cam_height=cam_height)
    if img is None:
        return None, None

    # 2) 如有內參，先取得 newK 並去畸變；同時將 corner/pocket 也轉到相同座標系
    newK = None
    if K is not None:
        h, w = img.shape[:2]
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
        img = cv2.undistort(img, K, D, None, newK)

    # 3) 依據影像當前的座標系（去畸變與否）建立 H 與 pockets
    pockets = _load_pockets([POCKETS_JSON, POCKET_JSON], K=K, D=D, newK=newK)
    if pockets is not None and not _validate_pockets(np.array(pockets, dtype=np.float32)):
        print("[WARN] pockets.json geometry looks invalid; falling back to corners for H")
        pockets = None
    H = _load_homography(CORNER_JSON, K=K, D=D, newK=newK, pockets=pockets)   # pixel → cm

    # 4) 偵測並轉換座標
    data, vis = _detect_and_convert(img, H, pockets)
    if show:
        cv2.imshow("YOLO",vis);cv2.waitKey(0);cv2.destroyAllWindows()

    out = SAVE_DIR/"cords.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out} ({len(data['balls'])} balls)")
    return str(out), data

# ═════════ 私用工具 ═════════

def _load_homography(corner_json: str, *, K=None, D=None, newK=None,
                     pockets: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Load 4 corners and build pixel→cm homography.

    If K/D are provided and the input image is undistorted with newK,
    the corner points are undistorted to the same pixel space before
    computing H. This keeps H consistent with detection coordinates.
    """
    # If pockets are available (6 points), prefer them to compute H,
    # since pocket centers lie on the inner playfield and give a more
    # robust mapping than possibly-outer corners.
    if pockets is not None and len(pockets) == 6:
        src = np.array(pockets, dtype=np.float32)
        if K is not None and D is not None and newK is not None:
            src_ud = cv2.undistortPoints(src.reshape(-1, 1, 2), K, D, P=newK)
            src = src_ud.reshape(-1, 2).astype(np.float32)
        dst = np.array(
            [
                [0, 0],
                [TABLE_W_CM, 0],
                [TABLE_W_CM, TABLE_H_CM],
                [0, TABLE_H_CM],
                [TABLE_W_CM / 2, 0],
                [TABLE_W_CM / 2, TABLE_H_CM],
            ],
            dtype=np.float32,
        )
        # Use RANSAC for robustness to one or two imprecise points
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=2.0)
        return H

    # Fallback to 4-corner homography
    with open(corner_json, 'r', encoding='utf-8') as f:
        c = json.load(f)
    src4 = np.array([
        [c['top_left']['x'],     c['top_left']['y']],
        [c['top_right']['x'],    c['top_right']['y']],
        [c['bottom_right']['x'], c['bottom_right']['y']],
        [c['bottom_left']['x'],  c['bottom_left']['y']],
    ], dtype=np.float32)
    if K is not None and D is not None and newK is not None:
        src_ud = cv2.undistortPoints(src4.reshape(-1, 1, 2), K, D, P=newK)
        src4 = src_ud.reshape(-1, 2).astype(np.float32)
    dst4 = np.array(
        [[0, 0], [TABLE_W_CM, 0], [TABLE_W_CM, TABLE_H_CM], [0, TABLE_H_CM]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src4, dst4)


def _validate_pockets(pts: np.ndarray) -> bool:
    """Basic geometry sanity check for pocket points in pixel space.

    Expects order: TL, TR, BR, BL, TM, BM. Returns True if plausible.
    """
    if pts.shape != (6, 2):
        return False
    tl, tr, br, bl, tm, bm = pts
    # Monotonicity roughly
    if not (tl[0] < tr[0] and bl[0] < br[0]):
        return False
    if not (tl[1] < bl[1] and tr[1] < br[1]):
        return False
    # Midpoints near the middle of top/bottom edges
    top_mid = (tl + tr) / 2.0
    bot_mid = (bl + br) / 2.0
    if np.linalg.norm(tm - top_mid) > 0.15 * np.linalg.norm(tr - tl):
        return False
    if np.linalg.norm(bm - bot_mid) > 0.15 * np.linalg.norm(br - bl):
        return False
    # Aspect ratio check (width/height ~ 1.96); allow generous tolerance
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_l = np.linalg.norm(bl - tl)
    h_r = np.linalg.norm(br - tr)
    w = 0.5 * (w_top + w_bot)
    h = 0.5 * (h_l + h_r)
    if w <= 1e-3 or h <= 1e-3:
        return False
    ratio = w / h
    if not (1.5 <= ratio <= 2.5):
        return False
    return True

def _load_pockets(px_json_paths: list[str], *, K=None, D=None, newK=None) -> list[tuple[float, float]] | None:
    """Load pockets (pixel) from JSON if available.

    Accepts formats:
    - dict with keys: top_left, top_right, bottom_right, bottom_left, top_mid, bottom_mid
    - list with 6 [x,y] pairs
    Returns list of 6 (x,y) tuples or None if no file found/invalid.
    """
    for p in px_json_paths:
        try:
            if not Path(p).exists():
                continue
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
            if isinstance(d, dict):
                names = [
                    "top_left", "top_right", "bottom_right",
                    "bottom_left", "top_mid", "bottom_mid",
                ]
                pts = []
                for k in names:
                    v = d.get(k)
                    if v is None or "x" not in v or "y" not in v:
                        raise ValueError("invalid pockets.json format")
                    pts.append((float(v["x"]), float(v["y"])))
                # Undistort pocket points if needed
                if K is not None and D is not None and newK is not None:
                    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
                    arr = cv2.undistortPoints(arr, K, D, P=newK)
                    pts = [tuple(map(float, p)) for p in arr.reshape(-1, 2)]
                return pts
            if isinstance(d, list) and len(d) == 6:
                pts = []
                for it in d:
                    if not isinstance(it, (list, tuple)) or len(it) != 2:
                        raise ValueError("invalid pockets list format")
                    pts.append((float(it[0]), float(it[1])))
                if K is not None and D is not None and newK is not None:
                    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
                    arr = cv2.undistortPoints(arr, K, D, P=newK)
                    pts = [tuple(map(float, p)) for p in arr.reshape(-1, 2)]
                return pts
        except Exception as e:
            print(f"[WARN] Failed to load pockets from {p}: {e}")
            continue
    return None

def _load_intrinsics(p:str):
    d=yaml.safe_load(open(p,'r'))
    M=d.get('camera_matrix',d.get('K'))
    if isinstance(M,dict):M=M['data']
    K=np.array(M,dtype=np.float32).reshape(3,3)
    D_=d.get('distortion_coefficients',d.get('dist_coeff',d.get('D')))
    if isinstance(D_,dict):D_=D_['data']
    D=np.array(D_,dtype=np.float32)
    return K,D

# --- 拍照工具 ---

def _snap(wait: int, preview: bool = True, *, cam_index: int | str = CAM_URL,
          cam_width: Optional[int] = None, cam_height: Optional[int] = None):
    cap = cv2.VideoCapture(cam_index)
    if cam_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    if cam_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    if not cap.isOpened():raise RuntimeError('Camera open fail')
    end=time.time()+wait
    img=None
    while time.time()<end:
        ok,frm=cap.read()
        if ok and preview:
            _draw_preview(frm,int(end-time.time())+1)
            if cv2.waitKey(30)&0xFF==27:
                cap.release();cv2.destroyAllWindows();return None
    ok,img=cap.read();cap.release();cv2.destroyAllWindows()
    if not ok:raise RuntimeError('Snap fail')
    return img

def _draw_preview(f,sec):
    cv2.putText(f,f"倒數 {sec}s",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
    cv2.imshow('Preview',f)

# --- 影像轉換 & 偵測 ---

def _undistort(img, K, D):
    # Deprecated by capture_balls flow (kept for backward compat if called elsewhere)
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
    return cv2.undistort(img, K, D, None, newK)


def _detect_and_convert(img: np.ndarray, H: np.ndarray, pockets: list[tuple[float, float]] | None):
    H_inv = np.linalg.inv(H)
    # 4 corner cm → pixel, also used for px-per-cm estimation
    cm_corners = np.array(
        [[0, 0], [TABLE_W_CM, 0], [TABLE_W_CM, TABLE_H_CM], [0, TABLE_H_CM]],
        dtype=np.float32,
    )
    px_corners = cv2.perspectiveTransform(cm_corners.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
    tl, tr, br, bl = px_corners

    # pockets: use provided (already undistorted if needed), else 4角+上下中點
    if pockets is None:
        pockets = [
            tuple(tl),
            tuple(tr),
            tuple(br),
            tuple(bl),
            tuple((tl + tr) / 2),
            tuple((bl + br) / 2),
        ]

    vis = img.copy()
    for px, py in pockets:
        cv2.circle(vis, (int(px), int(py)), POCKET_R_PX, (255, 0, 255), 2)

    model = YOLO(MODEL_PATH)
    r = model.predict(
        img,
        imgsz=640,
        conf=CONF_THRES,
        iou=IOU_THRES,
        agnostic_nms=True,
        verbose=False,
    )[0]

    # Collect detections sorted by confidence (high → low)
    boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
    classes    = r.boxes.cls.cpu().numpy()  if r.boxes is not None else np.empty((0,))
    confs      = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.empty((0,))
    order = np.argsort(-confs)

    # Greedy class-agnostic IoU suppression (keep highest-confidence when overlapping)
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
        area_b = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    keep_idx = []
    for i in order:
        a = boxes_xyxy[i]
        if all(_iou(a, boxes_xyxy[j]) <= IOU_THRES for j in keep_idx):
            keep_idx.append(i)
    dets = [
        (boxes_xyxy[i], classes[i], confs[i])
        for i in keep_idx
    ]

    # 影像→cm 轉換函式
    def px2cm(pt):
        x, y = pt
        v = H @ np.array([x, y, 1.0])
        return v[0] / v[2], v[1] / v[2]

    # px_per_cm for min‑sep
    table_px_w = np.linalg.norm(tr - tl)
    table_px_h = np.linalg.norm(bl - tl)
    min_sep_px = MIN_SEP_CM * (table_px_w / TABLE_W_CM + table_px_h / TABLE_H_CM) / 2

    balls, centers = [], []
    for box, cls, cf in dets:
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if any((cx - px) ** 2 + (cy - py) ** 2 <= POCKET_R_PX ** 2 for px, py in pockets):
            continue
        if any((cx - x0) ** 2 + (cy - y0) ** 2 < min_sep_px ** 2 for x0, y0 in centers):
            continue
        x_cm, y_cm = px2cm((cx, cy))
        balls.append(
            {
                "type": CLASS_NAMES[int(cls)],
                "conf": round(float(cf), 3),
                "x_cm": round(x_cm, 2),
                "y_cm": round(y_cm, 2),
            }
        )
        centers.append((cx, cy))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{CLASS_NAMES[int(cls)]} {float(cf):.2f}"
        cv2.putText(vis, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return {"timestamp": time.strftime("%Y%m%d_%H%M%S"), "balls": balls}, vis

# ═════════ CLI ═════════
if __name__=='__main__':
    capture_balls(wait_sec=3,show=True)
