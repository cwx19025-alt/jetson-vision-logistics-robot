import numpy as np
import cv2  # 用于保存图像（可选）
import matplotlib.pyplot as plt  # 用于预览图像

def generate_crossroad_image(
    img_size=(500, 500),  # 图像尺寸 (高度, 宽度)
    road_width=60,        # 道路宽度（像素）
    center=None           # 十字中心坐标 (y, x)，默认图像中心
):
    """
    生成标准十字路口黑白图像（道路白色255，背景黑色0）
    :param img_size: 图像尺寸，tuple (height, width)
    :param road_width: 道路宽度（像素），建议为偶数
    :param center: 十字中心 (y, x)，默认(img_size[0]//2, img_size[1]//2)
    :return: 黑白十字图像 (numpy数组，uint8类型)
    """
    # 1. 初始化全黑背景（0=黑色）
    crossroad = np.zeros(img_size, dtype=np.uint8)
    
    # 2. 确定十字中心
    if center is None:
        center_y = img_size[0] // 2
        center_x = img_size[1] // 2
    else:
        center_y, center_x = center
    
    # 3. 计算水平道路的上下边界（贯穿整幅图的水平路）
    h_road_top = max(0, center_y - road_width // 2)
    h_road_bottom = min(img_size[0], center_y + road_width // 2)
    # 绘制水平道路（设为255=白色）
    crossroad[h_road_top:h_road_bottom, :] = 255
    
    # 4. 计算垂直道路的左右边界（贯穿整幅图的垂直路）
    v_road_left = max(0, center_x - road_width // 2)
    v_road_right = min(img_size[1], center_x + road_width // 2)
    # 绘制垂直道路（设为255=白色）
    crossroad[:, v_road_left:v_road_right] = 255
    
    return crossroad

# ------------------- 测试生成 + 保存/预览 -------------------
if __name__ == "__main__":
    # 生成500x500的十字路口图像，道路宽度60像素
    cross_img = generate_crossroad_image(
        img_size=(500, 500),
        road_width=60
    )
    
    # 保存图像到本地（可选）
    cv2.imwrite("crossroad_black_white.png", cross_img)
    print("十字路口图像已保存为 crossroad_black_white.png")
    
    # 用matplotlib预览图像（可选）
    plt.figure(figsize=(6, 6))
    plt.imshow(cross_img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # 隐藏坐标轴
    plt.title("Standard Crossroad (White Road, Black Background)")
    plt.show()
    
    # 打印图像基本信息，验证像素值
    print(f"图像尺寸：{cross_img.shape}")
    print(f"道路像素值（白色）：{cross_img[250, 250]}")  # 中心像素
    print(f"背景像素值（黑色）：{cross_img[100, 100]}")  # 左上角背景