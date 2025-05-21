#!/usr/bin/env python3
"""
verify_corners.py  ─ 互動式調整版本
────────────────────────────────────────
用 corner.json 畫出 playfield 四角，可用滑鼠 **拖曳紅點** 微調。
按 **Space** 立即覆寫 corner.json 並顯示 "Saved!"。
按 **Esc / q** 離開（不存）。

操作提示
────────
▸ 滑鼠左鍵按住角點拖動
▸ Space：儲存（覆寫 corner.json）
▸ q / Esc：離開程式
"""

import cv2, json, numpy as np, os

IMG_PATH     = "photo_000.jpg"      # 原始照片（桌面）
CORNER_JSON  = "corner.json"    # 既有角點檔
POINT_COLOR  = (0, 0, 255)       # 紅色
LINE_COLOR   = (0, 255, 0)       # 綠色
FONT_COLOR   = (255, 0, 0)       # 藍色
RADIUS       = 10                # 點半徑
HIT_RADIUS   = 15                # 可點擊判定半徑

# ─── 讀影像與角點 ───
img = cv2.imread(IMG_PATH)
if img is None:
    raise SystemExit(f"找不到影像 {IMG_PATH}")

with open(CORNER_JSON, "r", encoding="utf-8") as f:
    c = json.load(f)
pts = np.array([
    [c["top_left"]["x"],     c["top_left"]["y"]],
    [c["top_right"]["x"],    c["top_right"]["y"]],
    [c["bottom_right"]["x"], c["bottom_right"]["y"]],
    [c["bottom_left"]["x"],  c["bottom_left"]["y"]],
], dtype=float)

selected_idx = None  # 目前被拖曳的點索引

# ─── 滑鼠事件回呼 ───

def mouse_cb(event, x, y, flags, _):
    global selected_idx, pts
    if event == cv2.EVENT_LBUTTONDOWN:
        # 找最近的點
        dists = np.linalg.norm(pts - np.array([x, y]), axis=1)
        idx = np.argmin(dists)
        if dists[idx] < HIT_RADIUS:
            selected_idx = idx
    elif event == cv2.EVENT_MOUSEMOVE and selected_idx is not None:
        pts[selected_idx] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_idx = None

# ─── 視覺化 ───

def draw():
    vis = img.copy()
    # 多邊形
    cv2.polylines(vis, [pts.astype(int)], True, LINE_COLOR, 3)
    # 角點
    for i, p in enumerate(pts):
        cv2.circle(vis, tuple(p.astype(int)), RADIUS, POINT_COLOR, -1)
        cv2.putText(vis, str(i), tuple(p.astype(int) + [10, -10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, FONT_COLOR, 2)
    return vis

cv2.namedWindow("Adjust corners")
cv2.setMouseCallback("Adjust corners", mouse_cb)
print("[INFO] 左鍵拖曳角點；Space = 儲存；q / Esc = 關閉")

while True:
    cv2.imshow("Adjust corners", draw())
    key = cv2.waitKey(20) & 0xFF
    if key in (27, ord('q')):   # Esc or q
        break
    elif key == 32:             # Space 保存
        data = {
            "top_left":     {"x": float(pts[0][0]), "y": float(pts[0][1])},
            "top_right":    {"x": float(pts[1][0]), "y": float(pts[1][1])},
            "bottom_right": {"x": float(pts[2][0]), "y": float(pts[2][1])},
            "bottom_left":  {"x": float(pts[3][0]), "y": float(pts[3][1])}
        }
        with open(CORNER_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {CORNER_JSON} 已更新")
        # 給使用者一個視覺提示：顯示文字 1 秒
        vis = draw()
        cv2.putText(vis, "Saved!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,255), 3)
        cv2.imshow("Adjust corners", vis)
        cv2.waitKey(800)

cv2.destroyAllWindows()
