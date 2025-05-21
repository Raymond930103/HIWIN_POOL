import cv2
import numpy as np
import yaml
import time

# === 讀取校正檔案 ===
with open("/Users/caiminhan/Projects/HIWIN_MAIN/main/vision/intrinsics.yaml") as f:
    data = yaml.safe_load(f)

K = np.array(data["K"], dtype=np.float32).reshape(3, 3)
D = np.array(data["D"], dtype=np.float32)

# === 開啟攝影機 ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# === 初始化 undistort 參數（需先取得影像尺寸） ===
ret, frame = cap.read()
if not ret:
    print("Error: Failed to grab initial frame.")
    cap.release()
    exit()

h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)

# === 主迴圈 ===
img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # 去畸變
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Undistorted Feed", undistorted)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按 q 離開
        break
    elif key == ord(' '):  # 按空白鍵拍照
        filename = f"photo_{img_count:03d}.jpg"
        cv2.imwrite(filename, undistorted)
        print(f"Saved: {filename}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()
