# yoloball_util.py
import cv2, json, time, numpy as np
from pathlib import Path
from ultralytics import YOLO

# ========= 共用參數（依需要修改 / 外部呼叫可覆寫） =========
JSON_CORNERS = "main/vision/corner.json"   # 4 角座標
MODEL_PATH   = "main/vision/best2.pt"      # YOLO 權重
CAM_URL      = 0                           # 攝影機 ID / rtsp
CLASS_NAMES  = ['8','10','2','15','7','14','6','13','5','12','4','11','3','0','1','9']
SCALE_PX_PER_CM = 22.11                    # 1 cm ≈ 22.11 px
CONF_THRES   = 0.25                        # YOLO 閾值
SAVE_DIR     = Path("main/vision/captured_json")       # JSON 目錄
POCKET_RADIUS_PX = 50                     # 口袋半徑 (px)

# ---------- 載入角點並計算 Homography（程式起始時做一次即可） ----------
with open(JSON_CORNERS, "r", encoding="utf-8") as f:
    d = json.load(f)
_src = np.float32([
    [d["top_left"]["x"],     d["top_left"]["y"]],
    [d["top_right"]["x"],    d["top_right"]["y"]],
    [d["bottom_right"]["x"], d["bottom_right"]["y"]],
    [d["bottom_left"]["x"],  d["bottom_left"]["y"]],
])
w_top,w_bot = np.linalg.norm(_src[1]-_src[0]), np.linalg.norm(_src[2]-_src[3])
h_l ,h_r    = np.linalg.norm(_src[3]-_src[0]), np.linalg.norm(_src[2]-_src[1])
DST_W, DST_H = int(max(w_top,w_bot)), int(max(h_l,h_r))
_dst = np.float32([[0,0],[DST_W-1,0],[DST_W-1,DST_H-1],[0,DST_H-1]])
H,_ = cv2.findHomography(_src, _dst)

# 六個球袋圓心（如需微調，直接改這裡）
POCKETS = [
    (50, 50),
    (DST_W-50, 50),
    (DST_W-50, DST_H-50),
    (50, DST_H-50),
    (DST_W//2, 30),
    (DST_W//2, DST_H-30)
]

# ---------- 主函式 ----------
def capture_balls(wait_sec: int = 3,
                  model_path: str = MODEL_PATH,
                  cam_url = CAM_URL,
                  conf_thres: float = CONF_THRES,
                  save_dir: Path = SAVE_DIR):
    """
    擷取一張桌面影像 → 偵測球 → 儲存 JSON
    回傳 (json_path, json_data)
    """
    # 1) 初始化
    model = YOLO(model_path)
    cap   = cv2.VideoCapture(cam_url)
    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機")

    print(f"[Info] 將於 {wait_sec} 秒後拍攝…")
    time.sleep(wait_sec)

    # 2) 取單張影像
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("攝影機擷取失敗")

    # 3) Homography 拉正
    warped = cv2.warpPerspective(frame, H, (DST_W, DST_H))

    # 4) YOLO 推論
    res = model.predict(warped, imgsz=640, conf=conf_thres, verbose=False)[0]

    # 5) 過濾球袋中的框
    kept = []
    for box, cls, cf in zip(res.boxes.xyxy.cpu(),
                            res.boxes.cls.cpu(),
                            res.boxes.conf.cpu()):
        x1,y1,x2,y2 = map(int, box)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        in_pocket = any((cx-px)**2 + (cy-py)**2 <= POCKET_RADIUS_PX**2
                        for px,py in POCKETS)
        if not in_pocket:
            kept.append((x1,y1,x2,y2,int(cls),float(cf),cx,cy))

    # 6) 組成輸出 JSON
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
    # 測試用：拍攝一張影像並儲存 JSON
    capture_balls(wait_sec=3, model_path=MODEL_PATH, cam_url=CAM_URL,
                  conf_thres=CONF_THRES, save_dir=SAVE_DIR)