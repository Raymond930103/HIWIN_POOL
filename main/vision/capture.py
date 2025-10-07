import cv2
import os
import yaml
import numpy as np

# 設定儲存影像的資料夾
save_folder = "captured_images"
os.makedirs(save_folder, exist_ok=True)  # 確保資料夾存在

# 找出下一個可用的檔名
def get_next_filename(folder):
    existing_files = [f for f in os.listdir(folder) if f.startswith("table") and f.endswith(".jpg")]
    
    if not existing_files:
        return "table1.jpg"  # 如果沒有任何符合格式的檔案，從 table1.jpg 開始

    existing_numbers = []
    for file in existing_files:
        try:
            # 找出 table 和 .jpg 中間的數字
            num_str = file[5:-4]
            num = int(num_str)
            existing_numbers.append(num)
        except ValueError:
            continue  # 略過不符合格式的檔案

    next_number = max(existing_numbers, default=0) + 1
    return f"table{next_number}.jpg"

# 初始化攝影機並拍攝單張照片
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


def capture(*, cam_index: int = 0, intrinsics_path: str | None = None, undistort: bool = True):
    camera = cv2.VideoCapture(cam_index)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    camera.set(cv2.CAP_PROP_EXPOSURE,-13)
    if not camera.isOpened():
        print("無法開啟攝影機")
        return None

    ret, frame = camera.read()
    if not ret:
        print("無法讀取影像")
        camera.release()
        return None

    if undistort:
        # Auto-pick default intrinsics if not provided
        if intrinsics_path is None and os.path.exists("main/vision/intrinsics.yaml"):
            intrinsics_path = "main/vision/intrinsics.yaml"
        if intrinsics_path and os.path.exists(intrinsics_path):
            try:
                K, D = _load_intrinsics(intrinsics_path)
                h, w = frame.shape[:2]
                newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
                frame = cv2.undistort(frame, K, D, None, newK)
            except Exception as e:
                print(f"[WARN] 去畸變失敗（略過）: {e}")

    img_filename = get_next_filename(save_folder)
    img_path = os.path.join(save_folder, img_filename)
    cv2.imwrite(img_path, frame)

    camera.release()
    cv2.destroyAllWindows()

    print(f"影像已儲存至 {img_path}")
    return img_path  # 回傳影像路徑


# 測試
if __name__ == '__main__':
    capture()
