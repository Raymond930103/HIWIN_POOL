#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 撞球偵測（Base = 球桌左上角）
───────────────────────────────────────────────────
‣ 使用 corner.json → Homography H(pixel→cm)，Base 原點 = 左上角 (0,0)。
‣ 偵測球心後直接輸出 Base‑XY (cm)。
‣ 修正袋口像素座標計算錯誤：改用 **H⁻¹(cm→pixel)** 反推四角。
"""
from __future__ import annotations

import cv2, json, time, yaml, numpy as np
from pathlib import Path
from typing import Tuple, List
from ultralytics import YOLO

# === 參數 ===
CAM_URL     = 0
SAVE_DIR    = Path("captures_json"); SAVE_DIR.mkdir(exist_ok=True)
CONF_THRES  = 0.10
POCKET_R_PX = 50
TABLE_W_CM  = 73.5
TABLE_H_CM  = 37.5
MIN_SEP_CM  = 1.0
MODEL_PATH  = "main/vision/best2.pt"
CLASS_NAMES = ['2','2','2','3','3','14','6','3','5','2','4','3','3','0','1','1']
CORNER_JSON = "main/vision/corner.json"

# ═════════ 公開 API ═════════

def capture_balls(*, wait_sec:int=3, show:bool=False, intrinsics_path:str|None=None
                  ) -> Tuple[str|None, dict|None]:
    """拍照→偵測→座標轉換→JSON；Esc 取消回 (None,None)"""

    H = _load_homography(CORNER_JSON)   # pixel → cm
    K=D=None
    if intrinsics_path:
        K,D=_load_intrinsics(intrinsics_path)

    img=_snap(wait_sec)
    if img is None: return None,None
    if K is not None:
        img=_undistort(img,K,D)

    data,vis=_detect_and_convert(img,H)
    if show:
        cv2.imshow("YOLO",vis);cv2.waitKey(0);cv2.destroyAllWindows()

    out=SAVE_DIR/"cords.json"
    with open(out,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
    print(f"[Saved] {out} ({len(data['balls'])} balls)")
    return str(out),data

# ═════════ 私用工具 ═════════

def _load_homography(corner_json:str)->np.ndarray:
    with open(corner_json,'r',encoding='utf-8') as f:
        c=json.load(f)
    src=np.array([
        [c['top_left']['x'],     c['top_left']['y']],
        [c['top_right']['x'],    c['top_right']['y']],
        [c['bottom_right']['x'], c['bottom_right']['y']],
        [c['bottom_left']['x'],  c['bottom_left']['y']],
    ],dtype=np.float32)
    dst=np.array([[0,0],[TABLE_W_CM,0],[TABLE_W_CM,TABLE_H_CM],[0,TABLE_H_CM]],dtype=np.float32)
    return cv2.getPerspectiveTransform(src,dst)   # 3×3 pixel→cm

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

def _snap(wait:int):
    cap=cv2.VideoCapture(CAM_URL)
    if not cap.isOpened():raise RuntimeError('Camera open fail')
    end=time.time()+wait
    img=None
    while time.time()<end:
        ok,frm=cap.read()
        if ok:
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

def _undistort(img,K,D):
    h,w=img.shape[:2]
    newK,_=cv2.getOptimalNewCameraMatrix(K,D,(w,h),0)
    return cv2.undistort(img,K,D,None,newK)


def _detect_and_convert(img:np.ndarray,H:np.ndarray):
    H_inv=np.linalg.inv(H)
    # 4 corner cm → pixel
    cm_corners=np.array([[0,0],[TABLE_W_CM,0],[TABLE_W_CM,TABLE_H_CM],[0,TABLE_H_CM]],dtype=np.float32)
    px_corners=cv2.perspectiveTransform(cm_corners.reshape(-1,1,2),H_inv).reshape(-1,2)
    tl,tr,br,bl=px_corners

    # pockets: 4角+2邊中點
    pockets=[tuple(tl),tuple(tr),tuple(br),tuple(bl),tuple((tl+tr)/2),tuple((bl+br)/2)]

    vis=img.copy()
    for px,py in pockets:
        cv2.circle(vis,(int(px),int(py)),POCKET_R_PX,(255,0,255),2)

    model=YOLO(MODEL_PATH)
    r=model.predict(img,imgsz=640,conf=CONF_THRES,verbose=False)[0]
    dets=sorted(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()),
                key=lambda x:float(x[2]), reverse=True)

    # 影像→cm 轉換函式
    def px2cm(pt):
        x,y=pt; v=H @ np.array([x,y,1.0]); return v[0]/v[2], v[1]/v[2]

    # px_per_cm for min‑sep
    table_px_w=np.linalg.norm(tr-tl)
    table_px_h=np.linalg.norm(bl-tl)
    min_sep_px=MIN_SEP_CM*(table_px_w/TABLE_W_CM+table_px_h/TABLE_H_CM)/2

    balls,centers=[],[]
    for box,cls,cf in dets:
        x1,y1,x2,y2=map(int,box); cx,cy=(x1+x2)//2,(y1+y2)//2
        if any((cx-px)**2+(cy-py)**2<=POCKET_R_PX**2 for px,py in pockets):
            continue
        if any((cx-x0)**2+(cy-y0)**2<min_sep_px**2 for x0,y0 in centers):
            continue
        x_cm,y_cm=px2cm((cx,cy))
        balls.append({"type":CLASS_NAMES[int(cls)],"conf":round(float(cf),3),"x_cm":round(x_cm,2),"y_cm":round(y_cm,2)})
        centers.append((cx,cy))
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.putText(vis,CLASS_NAMES[int(cls)],(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

    return {"timestamp":time.strftime("%Y%m%d_%H%M%S"),"balls":balls},vis

# ═════════ CLI ═════════
if __name__=='__main__':
    capture_balls(wait_sec=3,show=True)
