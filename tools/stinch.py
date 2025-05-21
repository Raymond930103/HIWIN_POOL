import cv2
import numpy as np

# 已知偏移量（單位：cm）
offset1_cm = 12.7264
offset2_cm = 12.4668

# 像素換算比例：1cm ≈ 16.62 px
scale = 16.62
offset1 = int(offset1_cm * scale)
offset2 = int(offset2_cm * scale)

# 讀取圖片
img1 = cv2.imread('./captured_images/table5.jpg')
img2 = cv2.imread('./captured_images/table6.jpg')
img3 = cv2.imread('./captured_images/table7.jpg')

# 確保圖片高度統一（這裡你圖片應該本來就一樣大）
h = 960
canvas_width = img1.shape[1] + offset1 + offset2
canvas = np.zeros((h, canvas_width, 3), dtype=np.uint8)

# 拼接圖片
canvas[:, 0:img1.shape[1]] = img1
canvas[:, offset1:offset1+img2.shape[1]] = np.maximum(canvas[:, offset1:offset1+img2.shape[1]], img2)
canvas[:, offset1+offset2:offset1+offset2+img3.shape[1]] = np.maximum(
    canvas[:, offset1+offset2:offset1+offset2+img3.shape[1]], img3)

# 儲存與顯示
cv2.imwrite('stitched_precise.jpg', canvas)
cv2.imshow('Stitched Result', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
