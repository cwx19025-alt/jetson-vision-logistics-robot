import cv2
import numpy as np
import os

# ========== 核心配置：修改为你的图片路径 ==========
# 图片所在文件夹绝对路径（无中文，无需处理编码）
IMG_DIR = r"C:\Users\cwx\Desktop\jetson orin nano\camera_1"
# 棋盘格内角点数量（根据你的棋盘格实际情况调整）
CHESSBOARD_W = 9  # 横向内角点
CHESSBOARD_H = 9  # 纵向内角点
# 棋盘格每个格子的实际尺寸（mm）
CHESSBOARD_SIZE = 18.1

# ========== 标定参数设置 ==========
# 亚像素角点检测停止准则
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 生成世界坐标系中的棋盘格点（三维，Z轴为0）
objp = np.zeros((CHESSBOARD_W * CHESSBOARD_H, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_W, 0:CHESSBOARD_H].T.reshape(-1, 2)
objp = objp * CHESSBOARD_SIZE  # 缩放为实际尺寸

# 存储标定用的点对
objpoints = []  # 世界坐标系的三维点
imgpoints = []  # 图像平面的二维点
img_size = None  # 存储图像尺寸（避免变量作用域问题）

# ========== 生成图片列表并读取 ==========
# 生成 img1.jpg 到 img15.jpg 的完整路径
image_paths = []
for i in range(1, 16):
    img_name = f"img{i}.jpg"
    full_path = os.path.join(IMG_DIR, img_name)
    image_paths.append(full_path)

# 遍历图片找角点
valid_img_count = 0
for idx, fname in enumerate(image_paths):
    # 读取图片
    img = cv2.imread(fname)
    if img is None:
        print(f"⚠️  跳过无法读取的图片：{fname}")
        continue
    
    # 获取图像尺寸（只记录一次）
    if img_size is None:
        img_size = img.shape[1::-1]  # (宽度, 高度)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(
        gray, (CHESSBOARD_W, CHESSBOARD_H), None
    )
    
    # 如果找到角点，进行亚像素优化并存储
    if ret:
        valid_img_count += 1
        print(f"✅  图片 {idx+1} ({fname}) 找到角点，序号：{valid_img_count}")
        
        # 亚像素角点优化
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 存储点对
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # 绘制角点并显示
        cv2.drawChessboardCorners(img, (CHESSBOARD_W, CHESSBOARD_H), corners, ret)
        cv2.namedWindow('Chessboard Corners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Chessboard Corners', 640, 480)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(200)  # 显示200ms

cv2.destroyAllWindows()

# ========== 相机标定 ==========
if len(objpoints) == 0:
    print("❌  错误：没有找到任何有效棋盘格角点，无法标定！")
else:
    print(f"\n📊  共找到 {valid_img_count} 张有效图片，开始标定...")
    
    # 执行标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    
    # 打印标定结果
    print("\n========== 标定结果 ==========")
    print(f"标定误差 (ret)：{ret:.6f}")
    print(f"\n内参矩阵 (mtx)：\n{mtx}")
    print(f"\n畸变系数 (dist)：\n{dist}")
    print(f"\n第一个图片的旋转向量 (rvecs[0])：\n{rvecs[0]}")
    print(f"\n第一个图片的平移向量 (tvecs[0])：\n{tvecs[0]}")
    
    # 计算最优新相机矩阵（用于畸变矫正）
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 0, (w, h)
    )
    print(f"\n最优新相机矩阵 (newcameramtx)：\n{newcameramtx}")
    
    # ========== 实时摄像头畸变矫正 ==========
    print("\n📹  打开摄像头进行实时矫正，按 'q' 退出并保存矫正后图片...")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌  无法打开摄像头！")
    else:
        while True:
            grabbed, frame = camera.read()
            if not grabbed:
                print("⚠️  无法读取摄像头画面")
                break
            
            # 畸变矫正（两种方法，选效率更高的remap）
            mapx, mapy = cv2.initUndistortRectifyMap(
                mtx, dist, None, newcameramtx, (w, h), 5
            )
            dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            
            # 裁剪矫正后的图像（可选）
            x, y, w_roi, h_roi = roi
            dst_cropped = dst[y:y+h_roi, x:x+w_roi]
            
            # 显示矫正后画面
            cv2.imshow('Undistorted Frame', dst_cropped)
            
            # 按q退出并保存图片
            if cv2.waitKey(1) & 0xFF == ord('q'):
                save_path = r"C:\Users\cwx\Desktop\jetson orin nano\undistorted_frame.jpg"
                cv2.imwrite(save_path, dst_cropped)
                print(f"\n✅  矫正后图片已保存至：{save_path}")
                break
        
        camera.release()
        cv2.destroyAllWindows()

print("\n🔚  标定程序结束")