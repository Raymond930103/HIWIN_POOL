import cv2
import os

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
def capture():
    camera = cv2.VideoCapture(0)
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
