import numpy as np
import cv2

# 生成直线通道图
def create_straight_road():
    h, w = 240, 320
    img = np.ones((h, w), dtype=np.uint8) * 255  # 白底

    # 画两条黑色边线（中间是路）
    left_line_x = 120
    right_line_x = 200

    for y in range(h):
        img[y, left_line_x] = 0   # 左边黑线
        img[y, right_line_x] = 0  # 右边黑线

    return img

# 保存
img_straight = create_straight_road()
cv2.imwrite(r"C:\Users\cwx\Desktop\jetson orin nano\track\straight_road.png", img_straight)
cv2.imshow("Straight Road", img_straight)
cv2.waitKey(0)