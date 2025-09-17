"""
將 YOLO 偵測 JSON (cm) 轉成 compute_shot() 輸入，
回傳 angle_deg + cue 座標，可選擇 --show 圖形化。

用法：
    python run_shot.py <json> [target_id|'min'] [--show]

參數說明
---------
<json>      ：YOLO 偵測結果路徑
[target_id] ：
    不輸入      → 取 conf 最高的非 0 號球
    整數 n      → 指定球號 n
    'min'       → 自動選擇除了 0 以外編號最小的球
--show      ：顯示圖形化路徑

此版本採 **作法 A**：
  ‑ 所有錯誤在 `plan_shot_from_json()` 內部捕捉並回傳 `None`，
  ‑ 呼叫端只需判斷是否為 `None`。
"""
import json
import argparse
from typing import Optional, Tuple, List, Union

# 允許作為套件 (import main.run_shot) 或腳本 (python main/run_shot.py) 執行
try:
    from .core.billiard_api import compute_shot         # 當作套件匯入時
    from . import gui as _gui_pkg  # 確保相對匯入生效
    import main.gui.visualize as visualize
except Exception:
    # 直接腳本執行時，將本檔所在資料夾加入 sys.path
    import os, sys
    sys.path.append(os.path.dirname(__file__))
    from core.billiard_api import compute_shot          # 腳本模式
    import gui.visualize as visualize                   # 腳本模式


# ──────────────── 工具 ────────────────

def cm2m(x_cm: float, y_cm: float) -> Tuple[float, float]:
    """cm → m"""
    return x_cm / 100.0, y_cm / 100.0


def plan_shot_from_json(
    json_path: str,
    target_id: Optional[Union[int, str]] = None,
    show: bool = False,
) -> Optional[Tuple[float, Tuple[float, float]]]:
    """讀取偵測結果並規劃擊球

    成功 → (angle_deg, cue_xy)
    失敗 → None（並印出錯誤訊息）
    """
    try:
        # --- 讀檔 ---
        with open(json_path, "r", encoding="utf-8") as f:
            raw_balls = json.load(f)["balls"]

        balls = [b for b in raw_balls if b["conf"] >= 0.30]
        if not balls:
            raise RuntimeError("JSON 內沒有信心值 ≥0.30 的球")

        # --- cue 球 ---
        cue_b = next(b for b in balls if b["type"] == "0")

        # --- 目標球邏輯 ---
        if target_id is None:
            # conf 最高
            tgt_b = max((b for b in balls if b["type"] != "0"), key=lambda b: b["conf"])
        elif target_id == "min":
            tgt_b = min((b for b in balls if b["type"] != "0"), key=lambda b: int(b["type"]))
        else:
            tgt_b = next((b for b in balls if b["type"] == str(target_id)), None)
            if tgt_b is None:
                raise RuntimeError(f"找不到球號 {target_id}")

        blk_bs: List[dict] = [b for b in balls if b not in (cue_b, tgt_b)]

        # --- cm → m ---
        # 支援兩種鍵名：舊版 (cx_cm/cy_cm) 與 新版 (x_cm/y_cm)
        def get_xy_cm(b: dict):
            if "cx_cm" in b and "cy_cm" in b:
                return b["cx_cm"], b["cy_cm"]
            elif "x_cm" in b and "y_cm" in b:
                return b["x_cm"], b["y_cm"]
            else:
                raise RuntimeError("ball item 缺少座標欄位 (cx_cm/cy_cm 或 x_cm/y_cm)")

        cx, cy = get_xy_cm(cue_b)
        tx, ty = get_xy_cm(tgt_b)
        cue_xy = cm2m(cx, cy)
        target = cm2m(tx, ty)
        blocks = [cm2m(*get_xy_cm(b)) for b in blk_bs]

        # --- 求解 ---
        info = compute_shot(cue_xy, target, blocks)
        if info is None:
            raise RuntimeError("無可行路徑 (compute_shot 回傳 None)")

        if show:
            visualize.show(cue_xy, target, blocks, info)

        return info["angle_deg"], cue_xy

    except Exception as e:
        print("[plan_shot] 失敗：", e)
        return None


# ──────────────── CLI ────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="YOLO 偵測結果 .json 路徑")
    ap.add_argument("id", nargs="?", help="目標球號；輸入 'min' 取最小球")
    ap.add_argument("--show", action="store_true", help="顯示圖形化路徑")
    args = ap.parse_args()

    # 解析目標參數
    if args.id is None:
        target_param: Optional[Union[int, str]] = None
    elif args.id.lower() == "min":
        target_param = "min"
    else:
        try:
            target_param = int(args.id)
        except ValueError:
            print(f"[run_shot] 提供的球號 {args.id!r} 不是整數，也不是 'min'")
            exit(1)

    # 呼叫函式 ─ 成功回 (angle, cue)；失敗回 None
    result = plan_shot_from_json(args.json, target_param, show=args.show)

    if result is None:
        print("→ None")
    else:
        angle_deg, cue_xy = result
        print(
            f"→ angle_deg = {angle_deg:.2f}°, cue = ({cue_xy[0]:.3f} m, {cue_xy[1]:.3f} m)"
        )
