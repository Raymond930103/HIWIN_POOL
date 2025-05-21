#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
billiard_capture_nocorner.py
----------------------------
相機已與桌面平行且填滿整張桌子，不使用 Homography。
capture_balls(countdown=3, show=False)：
    1) 倒數拍照
    2) HoughCircles 找球
    3) 像素 → cm (線性比例)
    4) 輸出 CORDS.json
"""

from __future__ import annotations
import cv2, json, time, numpy as np
from pathlib import Path
from typing import Tuple, Optional

# ======== 需自行設定 ========
TABLE_W_CM   = 73        # 桌面水平長度 (cm)  ← 換成你的
TABLE_H_CM   = 40        # 桌面垂直長度 (cm) ← 換成你的
CAM_URL      = 0           # 攝影機 ID / RTSP
SAVE_DIR     = Path(".")   # CORDS.json 儲存位置
POCKET_R_PX  = 70          # 口袋半徑 (px) 視畫面解析度調整
POCKET_OFFSET_PX = 75

# ---------- 顏色區段 & 球號對應 ----------
COLOR_RANGES = {
    "yellow": (( 20,  80,  80), ( 35, 255, 255)),
    "blue"  : (( 90,  80,  80), (120, 255, 255)),
    "red1"  : ((  0,  80,  80), ( 10, 255, 255)),
    "red2"  : ((170,  80,  80), (180, 255, 255)),
    "purple": ((130,  60,  60), (150, 255, 255)),
    "orange": (( 10, 100, 100), ( 20, 255, 255)),
    "green" : (( 40,  60,  60), ( 80, 255, 255)),
    "maroon": ((160,  70,  40), (175, 255, 180)),
    "black" : ((  0,   0,   0), (180, 255,  60)),
}
COLOR_TO_BALL = {
    "yellow": ("1","9"), "blue":("2","10"), "red":("3","11"),
    "purple":("4","12"), "orange":("5","13"), "green":("6","14"),
    "maroon":("7","15"), "black":("8",)
}

# ---------- 分色判號 ----------
def classify_ball(hsv_pixels: np.ndarray) -> Optional[str]:
    if hsv_pixels.size == 0: return None
    h,s,v = hsv_pixels.mean(axis=0)
    white_ratio = np.sum((hsv_pixels[:,2]>200)&(hsv_pixels[:,1]<25))/len(hsv_pixels)
    if white_ratio > 0.7: return "0"
    if v<70 and s<60: return "8"
    for name,(lo,hi) in COLOR_RANGES.items():
        base = "red" if name.startswith("red") else name
        l,u = np.array(lo),np.array(hi)
        hue_ok = l[0]<=h<=u[0] if l[0]<=u[0] else (h>=l[0] or h<=u[0])
        if not hue_ok or not(l[1]<=s<=u[1] and l[2]<=v<=u[2]): continue
        if base=="black": return "8"
        solid,stripe = COLOR_TO_BALL[base]
        return stripe if white_ratio>0.25 else solid
    return None

# ========= 主功能 =========
def capture_balls(countdown:int=3, show:bool=False):
    cap=cv2.VideoCapture(CAM_URL)
    if not cap.isOpened(): raise RuntimeError("無法開啟攝影機")

    # 先抓一張畫面，推算解析度與縮放比例
    ret, frame0 = cap.read()
    if not ret: raise RuntimeError("讀取影像失敗")
    H_img, W_img = frame0.shape[:2]
    scale_x = W_img / TABLE_W_CM      # px / cm
    scale_y = H_img / TABLE_H_CM

    # 口袋中心位置 (以像素為單位)
    pockets = [
    (POCKET_OFFSET_PX, POCKET_OFFSET_PX),                              # 左上
    (W_img - POCKET_OFFSET_PX, POCKET_OFFSET_PX),                      # 右上
    (W_img - POCKET_OFFSET_PX, H_img - POCKET_OFFSET_PX),              # 右下
    (POCKET_OFFSET_PX, H_img - POCKET_OFFSET_PX),                      # 左下
    (W_img // 2, POCKET_OFFSET_PX // 2),                               # 上側中袋
    (W_img // 2, H_img - POCKET_OFFSET_PX // 2)                        # 下側中袋
]


    # 倒數
    end=time.time()+countdown
    while time.time()<end:
        ok, prev = cap.read();   now = time.time()
        if not ok: continue
        sec = int(end-now)+1
        cv2.putText(prev,f"倒數 {sec}s",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
        cv2.imshow("Preview",prev)
        if cv2.waitKey(30)&0xFF==27:
            cap.release();cv2.destroyAllWindows();return None,None
    ok, img = cap.read(); cap.release(); cv2.destroyAllWindows()
    if not ok: raise RuntimeError("拍照失敗")

    # Hough 找球
    gray=cv2.medianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),5)
    circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.1,35,
                             param1=80,param2=25,minRadius=20,maxRadius=45)

    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    balls=[]
    if circles is not None:
        for cx,cy,r in np.round(circles[0]).astype(int):
            if any((cx-px)**2+(cy-py)**2<=POCKET_R_PX**2 for px,py in pockets):
                continue
            mask=np.zeros(gray.shape,np.uint8);cv2.circle(mask,(cx,cy),r,255,-1)
            ball_id=classify_ball(hsv[mask==255])
            if ball_id is None: continue
            x_cm = cx / scale_x
            y_cm = cy / scale_y
            if show:
                cv2.circle(img,(cx,cy),r,(0,255,255),2)
                cv2.putText(img,ball_id,(cx-r,cy-r),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            balls.append({
                "type":ball_id,"conf":1.0,
                "cx_cm":round(x_cm,2),"cy_cm":round(y_cm,2)
            })

    # 口袋視覺化
    if show:
        for px,py in pockets:
            cv2.circle(img,(px,py),POCKET_R_PX,(255,0,255),2)
        cv2.imshow("Result",img);cv2.waitKey(0);cv2.destroyAllWindows()

    # 儲存
    SAVE_DIR.mkdir(exist_ok=True)
    json_path = SAVE_DIR/"CORDS.json"
    out={"timestamp":time.strftime("%Y%m%d_%H%M%S"),"balls":balls}
    with open(json_path,"w",encoding="utf-8") as fp:
        json.dump(out,fp,ensure_ascii=False,indent=2)
    print(f"[Saved] {json_path} ({len(balls)} balls)")
    return str(json_path), out

# 示範執行
if __name__=="__main__":
    capture_balls(countdown=3, show=True)
