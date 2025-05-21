import cv2
import numpy as np

def stitch_images(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"好匹配點數量：{len(good)}")

    if len(good) < 4:
        print("匹配點太少，拼接失敗")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # warp 第二張圖到第一張圖的空間中
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 預估 warp 後最大畫布大小
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)
    all_corners = np.concatenate((np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2), warped_corners), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]

    H_translation = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # 平移

    result = cv2.warpPerspective(img2, H_translation @ H, (xmax - xmin, ymax - ymin))
    result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

    return result
img1 = cv2.imread('./captured_images/table5.jpg')
img2 = cv2.imread('./captured_images/table6.jpg')
img3 = cv2.imread('./captured_images/table7.jpg')

# 依序拼接
step1 = stitch_images(img1, img2)
if step1 is not None:
    final = stitch_images(step1, img3)
    if final is not None:
        cv2.imwrite("stitched_fixed.jpg", final)
        cv2.imshow("Result", final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("step2 拼接失敗")
else:
    print("step1 拼接失敗")
