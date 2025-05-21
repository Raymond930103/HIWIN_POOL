from ultralytics import YOLO
import cv2
import numpy as np
import glob

# 載入訓練好的模型
model = YOLO("best.pt")  # 替換成你的模型檔路徑

# 撞球的分類名稱（index 依訓練資料順序）
class_names = [
    "black_8", "blue_10", "blue_2", "dred_15", "dred_7", "green_14", "green_6",
    "orange_13", "orange_5", "purple_12", "purple_4", "red_11", "red_3",
    "white", "yellow_1", "yellow_9"
]

# 載入影像（改成你的圖片路徑）
image_path = "./captured_images/table7.jpg"
img = cv2.imread(image_path)
h, w = img.shape[:2]

# 執行 YOLO 推論
results = model(img)[0]

# 取出預測框
boxes = results.boxes
for i in range(len(boxes)):
    cls_id = int(boxes.cls[i])
    conf = float(boxes.conf[i])
    x1, y1, x2, y2 = map(int, boxes.xyxy[i])

    # 繪製框框與文字
    label = f"{class_names[cls_id]} {conf:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 顯示結果
cv2.imshow("Detected Balls", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
