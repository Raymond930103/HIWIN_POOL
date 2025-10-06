import cv2, json, time, numpy as np, yaml
from pathlib import Path
from ultralytics import YOLO

# ========= 共用參數（依需要修改 / 外部呼叫可覆寫） =========
JSON_CORNERS = "main/vision/corner.json"   # 4 角座標（未去畸變像素座標）
INTRINSICS   = "main/vision/intrinsics.yaml"  # 內參（可選；若存在就使用）
MODEL_PATH   = "main/vision/best2.pt"      # YOLO 權重
CAM_URL      = 0                           # 攝影機 ID / rtsp
CLASS_NAMES  = ['0','1','10','11','12','13','14','15','2','3','4','5','6','7','8','9']
SCALE_PX_PER_CM = 22.11                    # 1 cm ≈ 22.11 px（拉正圖上的像素/公分）
CONF_THRES   = 0.25                        # YOLO 閾值
SAVE_DIR     = Path("main/vision/captured_json")       # JSON 目錄
POCKET_RADIUS_PX = 50                     # 口袋半徑 (px)


def _load_intrinsics(p: str):
    with open(p, 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f)
    M = d.get('camera_matrix', d.get('K'))
    if isinstance(M, dict):
        M = M.get('data', M.get('vals'))
    K = np.array(M, dtype=np.float32).reshape(3, 3)
    D_ = d.get('distortion_coefficients', d.get('dist_coeff', d.get('D')))
    if isinstance(D_, dict):
        D_ = D_.get('data', D_.get('vals'))
    D = np.array(D_, dtype=np.float32).reshape(-1)
    return K, D


def _compute_homo_from_corners(corners_json: str, K=None, D=None, newK=None):
    # 讀 corner.json（一般為未去畸變像素座標）
    with open(corners_json, "r", encoding="utf-8") as f:
        c = json.load(f)
    src4 = np.float32([
        [c["top_left"]["x"],     c["top_left"]["y"]],
        [c["top_right"]["x"],    c["top_right"]["y"]],
        [c["bottom_right"]["x"], c["bottom_right"]["y"]],
        [c["bottom_left"]["x"],  c["bottom_left"]["y"]],
    ])
    # 若有內參，將角點轉到去畸變座標系（與去畸變影像一致）
    if K is not None and D is not None and newK is not None:
        pts_ud = cv2.undistortPoints(src4.reshape(-1, 1, 2), K, D, P=newK)
        src4 = pts_ud.reshape(-1, 2).astype(np.float32)
    w_top, w_bot = np.linalg.norm(src4[1]-src4[0]), np.linalg.norm(src4[2]-src4[3])
    h_l, h_r    = np.linalg.norm(src4[3]-src4[0]), np.linalg.norm(src4[2]-src4[1])
    dst_w, dst_h = int(max(w_top, w_bot)), int(max(h_l, h_r))
    dst4 = np.float32([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]])
    H, _ = cv2.findHomography(src4, dst4)
    return H, dst_w, dst_h


def capture_balls(wait_sec: int = 3,
                  model_path: str = MODEL_PATH,
                  cam_url = CAM_URL,
                  conf_thres: float = CONF_THRES,
                  save_dir: Path = SAVE_DIR,
                  intrinsics_path: str | None = None,
                  corner_json: str = JSON_CORNERS):
    """
    擷取一張桌面影像 → （可選）去畸變 → Homography 拉正 → 偵測球 → 儲存 JSON
    回傳 (json_path, json_data)
    """
    model = YOLO(model_path)
    cap   = cv2.VideoCapture(cam_url)
    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機")

    print(f"[Info] 將於 {wait_sec} 秒後拍攝…")
    time.sleep(wait_sec)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("攝影機擷取失敗")

    h, w = frame.shape[:2]

    # Optional undistort
    K = D = newK = None
    if intrinsics_path is None and Path(INTRINSICS).exists():
        intrinsics_path = INTRINSICS
    if intrinsics_path and Path(intrinsics_path).exists():
        try:
            K, D = _load_intrinsics(intrinsics_path)
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
            frame = cv2.undistort(frame, K, D, None, newK)
            print(f"[Undistort] Using {intrinsics_path}")
        except Exception as e:
            print(f"[WARN] 內參讀取失敗，跳過去畸變: {e}")

    # Homography （角點自動轉到對應的座標系）
    H, DST_W, DST_H = _compute_homo_from_corners(corner_json, K, D, newK)
    warped = cv2.warpPerspective(frame, H, (DST_W, DST_H))

    res = model.predict(warped, imgsz=640, conf=conf_thres, verbose=False)[0]

    # 口袋定義（在拉正影像上）
    POCKETS = [
        (50, 50),
        (DST_W-50, 50),
        (DST_W-50, DST_H-50),
        (50, DST_H-50),
        (DST_W//2, 30),
        (DST_W//2, DST_H-30)
    ]

    kept = []
    for box, cls, cf in zip(res.boxes.xyxy.cpu(),
                            res.boxes.cls.cpu(),
                            res.boxes.conf.cpu()):
        x1,y1,x2,y2 = map(int, box)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        in_pocket = any((cx-px)**2 + (cy-py)**2 <= POCKET_RADIUS_PX**2 for px,py in POCKETS)
        if not in_pocket:
            kept.append((x1,y1,x2,y2,int(cls),float(cf),cx,cy))

    ts  = time.strftime("%Y%m%d_%H%M%S")
    data = {"timestamp": ts, "balls":[]}
    for x1,y1,x2,y2,cls,cf,cx,cy in kept:
        data["balls"].append({
            "type" : CLASS_NAMES[cls],
            "conf" : round(cf,3),
            "cx_cm": round(cx / SCALE_PX_PER_CM, 2),
            "cy_cm": round(cy / SCALE_PX_PER_CM, 2),
            "bbox_px": [x1,y1,x2,y2]
        })

    save_dir.mkdir(exist_ok=True)
    json_path = save_dir / f"{ts}.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)

    print(f"[Saved] {json_path} ({len(data['balls'])} balls)")
    return str(json_path), data


if __name__ == "__main__":
    capture_balls()
