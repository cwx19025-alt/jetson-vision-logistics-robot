import cv2
import numpy as np
import math
import time   # 可选，用于调试延时

# ---------- 辅助函数 ----------
def draw_pts_black_white(w, h, pts, thickness=2):
    """创建黑白图像并绘制点集（白色）"""
    black_img = np.zeros((h, w, 3), dtype=np.uint8)
    if len(pts) == 0:
        return black_img
    white_color = (255, 255, 255)
    for pt in pts:
        x = int(round(pt[0]))
        y = int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(black_img, (x, y), thickness, white_color, -1)
    return black_img

# ---------- 左侧边缘检测类 ----------
class LeftEdgeDetector:
    LINE_BLUR_KERNEL = 3
    def __init__(self, frame, debug=False):
        self.stop = False
        self.debug = debug    
        self.frame = frame.copy()          # 保留原始帧副本用于调试绘制
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        self.dir_front = np.array([[0,-1],[1,0],[0,1],[-1,0]], dtype=np.int32)   # 上右下左
        self.dir_left  = np.array([[-1,0],[0,-1],[1,0],[0,1]], dtype=np.int32)   # 左上下右（相对当前方向）
        self.h, self.w = self.gray.shape[:2]
        self.edge_pts_filtered = []
        self.blur_pts = []
        self.resampled_pts = []

    def binary_value(self, frame, x, y, block_size):
        """局部自适应二值化：若中心像素亮度 ≥ 邻域均值则返回255，否则0"""
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return 0
        pixel = frame[int(y), int(x)]
        half = block_size // 2
        x0 = int(max(0, x - half))
        x1 = int(min(self.w, x + half))
        y0 = int(max(0, y - half))
        y1 = int(min(self.h, y + half))
        block = frame[y0:y1+1, x0:x1+1]
        # return 255 if np.mean(block) <= pixel else 0
        return 255 if np.mean(block) > pixel else 0

    def find_left_edge(self, block_size=7, start_x=None, start_y=None):
        """从底部中心开始，遵循左手规则追踪左侧边缘"""
        if start_x is None:
            start_x = int(self.w / 2)
        if start_y is None:
            start_y = int(self.h - 1)
        
        assert len(self.gray.shape) == 2
        assert block_size % 2 == 1
        assert 0 <= start_x < self.w and 0 <= start_y < self.h

        self.edge_pts = []
        dir_idx = 0          # 初始方向：向上
        turn = 0

        # 起始点处理
        if self.binary_value(self.gray, start_x, start_y, block_size) == 0:
            self.edge_pts.append([start_x, start_y])

        print(f'起始坐标: ({start_x}, {start_y})')

        while start_x >= 0 and start_y >= 0 and start_x < self.w and start_y <= self.h and turn < 4 and not self.stop:
            if start_y == 0:   # 到达图像顶部
                break

            # 计算前方和左侧点坐标
            x_front = start_x + self.dir_front[dir_idx][0]
            y_front = start_y + self.dir_front[dir_idx][1]
            x_left  = start_x + self.dir_left[dir_idx][0]
            y_left  = start_y + self.dir_left[dir_idx][1]

            # 闭环检测（回到起点）
            if len(self.edge_pts) >= 40 and start_x == self.edge_pts[0][0] and start_y == self.edge_pts[0][1]:
                break

            left_val = self.binary_value(self.gray, x_left, y_left, block_size)
            front_val = self.binary_value(self.gray, x_front, y_front, block_size)

            if left_val == 255:                     # 左侧为白色 → 左转并前进
                start_x, start_y = x_left, y_left
                self.edge_pts.append([start_x, start_y])
                dir_idx = (dir_idx - 1) % 4
                turn = 0
                if self.debug:
                    cv2.circle(self.frame, (int(start_x), int(start_y)), 1, (255,255,0), -1)
                    print("左转并前进")
            elif front_val == 0:                    # 前方为黑色 → 右转（不移动）
                dir_idx = (dir_idx + 1) % 4
                turn += 1
                if self.debug:
                    print("右转")
            else:                                   # 前方为白色 → 前进
                start_x, start_y = x_front, y_front
                self.edge_pts.append([start_x, start_y])
                turn = 0
                if self.debug:
                    cv2.circle(self.frame, (int(start_x), int(start_y)), 1, (255,255,0), -1)
                    print("前进")

            if self.debug:
                # 显示当前方向（调试用）
                direction_names = ["向上", "向右", "向下", "向左"]
                print(f"方向: {direction_names[dir_idx]}, 坐标: ({start_x}, {start_y})")
                cv2.imshow('debug', self.frame)
                cv2.waitKey(1)   # 按任意键继续（若需要单步可改为0）

        if turn == 4:
            print("警告: 未找到左边缘，可能陷入死循环")

    def filter_edge_pts(self, threshold=5):
        """剔除距离过近的点"""
        if len(self.edge_pts) < 2:
            return
        filtered = [self.edge_pts[0]]
        for i in range(1, len(self.edge_pts)):
            if np.linalg.norm(np.array(self.edge_pts[i]) - np.array(filtered[-1])) > threshold:
                filtered.append(self.edge_pts[i])
        self.edge_pts_filtered = filtered

    def blur_points(self):
        """滑动窗口平均平滑"""
        if len(self.edge_pts_filtered) < self.LINE_BLUR_KERNEL:
            return
        half = self.LINE_BLUR_KERNEL // 2
        blur = []
        for i in range(len(self.edge_pts_filtered)):
            start = max(0, i - half)
            end = min(len(self.edge_pts_filtered), i + half + 1)
            window = self.edge_pts_filtered[start:end]
            mean_x = sum(p[0] for p in window) / len(window)
            mean_y = sum(p[1] for p in window) / len(window)
            blur.append([mean_x, mean_y])
        self.blur_pts = blur

    def resample_points(self, sample_dist=5.0):
        """等距重采样"""
        if len(self.blur_pts) < 2:
            return
        cum_dist = [0]
        for i in range(1, len(self.blur_pts)):
            cum_dist.append(cum_dist[-1] + np.linalg.norm(np.array(self.blur_pts[i]) - np.array(self.blur_pts[i-1])))
        total = cum_dist[-1]
        resampled = []
        cur = 0
        idx = 0
        while cur <= total:
            while idx < len(self.blur_pts)-1 and cur > cum_dist[idx+1]:
                idx += 1
            if idx >= len(self.blur_pts)-1:
                resampled.append(self.blur_pts[-1].copy())
                break
            t = (cur - cum_dist[idx]) / (cum_dist[idx+1] - cum_dist[idx])
            x = self.blur_pts[idx][0] + t * (self.blur_pts[idx+1][0] - self.blur_pts[idx][0])
            y = self.blur_pts[idx][1] + t * (self.blur_pts[idx+1][1] - self.blur_pts[idx][1])
            resampled.append([x, y])
            cur += sample_dist
        self.resampled_pts = resampled

    # 其他方法（local_angle_points, nms_angle, detect_L_edge, fit_bottom_angle）可自行补充
    # 此处省略以保持简洁，但可参考原始代码添加

# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 1. 读取图像（请替换为实际路径）
    image_path = r"C:\Users\cwx\Desktop\jetson orin nano\saved_frames2\frame_20260415_202832.jpg"
    frame = cv2.imread(image_path)
    if frame is None:
        print("无法读取图像，请检查路径")
        exit()

    # 2. 创建检测器，开启调试模式（会显示追踪过程）
    detector = LeftEdgeDetector(frame, debug=True)

    # 3. 执行边缘追踪
    detector.find_left_edge(block_size=7,start_x=100)   # 可调整block_size

    # 4. 后处理（可选）
    detector.filter_edge_pts(threshold=5)
    detector.blur_points()
    detector.resample_points(sample_dist=5.0)

    # 5. 可视化结果
    # 绘制原始追踪点（红色）
    vis_img = frame.copy()
    for pt in detector.edge_pts:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
    # 绘制滤波后的点（绿色）
    for pt in detector.edge_pts_filtered:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
    # 绘制重采样后的点（蓝色）
    for pt in detector.resampled_pts:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)

    cv2.imshow("Result", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6. 输出点集
    print(f"原始点数: {len(detector.edge_pts)}")
    print(f"滤波后点数: {len(detector.edge_pts_filtered)}")
    print(f"重采样点数: {len(detector.resampled_pts)}")