import time
import cv2
import numpy as np
import struct
import socket
import subprocess
import os
import sys


# ==================== v4l2-ctl 曝光和白平衡配置（参考 1231234.py） ====================
def init_video_exposure(real_dev, exposure_time_absolute):
    """配置摄像头曝光：关闭自动曝光，设置固定曝光时间"""
    cmd_list = [
        f"v4l2-ctl -d {real_dev} -c auto_exposure=1",
        f"v4l2-ctl -d {real_dev} -c exposure_time_absolute={exposure_time_absolute}"
    ]
    for cmd in cmd_list:
        print(f"\n==> 执行命令: {cmd}")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"✅ 命令执行成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ 命令执行失败！错误码: {e.returncode}, 错误信息: {e.stderr.strip()}")


def init_video_white_balance(real_dev, white_balance_temperature):
    """配置摄像头白平衡：关闭自动白平衡，设置固定色温"""
    cmd_list = [
        f"v4l2-ctl -d {real_dev} -c white_balance_automatic=0",
        f"v4l2-ctl -d {real_dev} -c white_balance_temperature={white_balance_temperature}"
    ]
    for cmd in cmd_list:
        print(f"\n==> 执行命令: {cmd}")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"✅ 命令执行成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ 命令执行失败！错误码: {e.returncode}, 错误信息: {e.stderr.strip()}")


# ==================== 摄像头参数配置（封装为函数，方便重用） ====================
def setup_camera(cap, device_index=0, 
                 img_width=640, img_height=480, fps=30,
                 exposure_time=360, white_balance_temperature=4900):
    """
    打开摄像头并配置分辨率、FPS、曝光和白平衡参数。
    返回 (cap, real_dev) 或退出程序。
    """
    # 打开摄像头
    real_dev = f"/dev/video{device_index}"
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 /dev/video{device_index}")
        sys.exit(1)

    # 设置分辨率和FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"✅ 摄像头 /dev/video{device_index} 打开成功")
    print(f"   分辨率: {img_width}x{img_height}, FPS: {fps}")

    # 配置曝光和白平衡（参考 1231234.py）
    init_video_exposure(real_dev, exposure_time)
    init_video_white_balance(real_dev, white_balance_temperature)

    return cap, real_dev


# ==================== 帧率控制器（平滑发送帧率） ====================
class FrameRateController:
    """帧率控制器，确保发送帧率不超过目标FPS"""
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_send_time = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0.0

    def wait(self):
        """等待到下一帧的发送时间"""
        current_time = time.time()
        elapsed = current_time - self.last_send_time
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
        self.last_send_time = time.time()

        # 统计FPS（每30帧打印一次）
        self.frame_count += 1
        if self.frame_count >= 30:
            real_elapsed = time.time() - self.fps_start_time
            self.avg_fps = self.frame_count / real_elapsed if real_elapsed > 0 else 0
            print(f"[FPS] 发送帧率: {self.avg_fps:.2f}")
            self.frame_count = 0
            self.fps_start_time = time.time()


# ==================== 客户端类（优化：增加重连机制、帧率控制） ====================
class Client:
    def __init__(self, ip='192.168.1.100', port=8080,
                 max_retries=-1, retry_interval=2):
        """
        :param ip: 服务端IP
        :param port: 服务端端口
        :param max_retries: 最大重连次数，-1表示无限重试
        :param retry_interval: 重连间隔（秒）
        """
        self.ip = ip
        self.port = port
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.client = None
        self.is_connected = False

        # 连接
        self.connect()

    def connect(self):
        """连接到服务端，支持重试"""
        retries = 0
        while self.client is None:
            if self.max_retries != -1 and retries >= self.max_retries:
                print(f"❌ 重连次数已达上限 ({self.max_retries})，退出程序")
                sys.exit(1)

            try:
                print(f"尝试连接 {self.ip}:{self.port} (第{retries + 1}次)")
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.settimeout(10)
                client.connect((self.ip, self.port))
                self.client = client
                self.is_connected = True
                print(f"✅ 连接成功 {self.ip}:{self.port}")
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                retries += 1
                print(f"❌ 连接失败：{e}")
                print(f"   等待 {self.retry_interval} 秒后重试...")
                time.sleep(self.retry_interval)

    def reconnect(self):
        """断开当前连接并重新连接"""
        self.close()
        self.client = None
        self.is_connected = False
        print("🔄 正在重新连接...")
        self.connect()

    def close(self):
        """关闭当前连接"""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

    def send_frame(self, frame, encode_format='.jpg', quality=85):
        """
        发送一帧图像到服务端（优化：支持多种编码格式、重连机制）
        
        :param frame: OpenCV图像帧 (BGR)
        :param encode_format: 编码格式 '.jpg' 或 '.png'
        :param quality: JPEG质量（0-100），仅对JPEG有效；PNG时为压缩级别
        :return: True=发送成功, False=发送失败
        """
        if not self.is_connected or self.client is None:
            print("❌ 未连接到服务端")
            return False

        try:
            # 编码图像
            if encode_format == '.png':
                # PNG无损编码
                params = [int(cv2.IMWRITE_PNG_COMPRESSION), quality]
            else:
                # JPEG有损编码（默认）
                params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

            encoded = cv2.imencode(encode_format, frame, params)
            if encoded is None or encoded[0] is None:
                print("❌ 图像编码失败")
                return False

            data = encoded[1].tobytes()
            length = len(data)

            # 打包长度头（4字节大端序）
            length_header = struct.pack('>I', length)

            # 发送长度头 + 图像数据
            self.client.sendall(length_header)
            self.client.sendall(data)
            return True

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as e:
            print(f"❌ 发送帧失败（连接断开）：{e}")
            self.is_connected = False
            # 尝试重连
            self.reconnect()
            return False

        except Exception as e:
            print(f"❌ 发送帧异常：{e}")
            return False


# ==================== 主程序 ====================
if __name__ == '__main__':
    # ====== 可配置参数（参考 1231234.py 的默认值） ======
    SERVER_IP = '192.168.1.100'
    SERVER_PORT = 8080
    DEVICE_INDEX = 0
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    TARGET_FPS = 30
    EXPOSURE_TIME = 360          # 参考 1231234.py exposure_time_yolo = 360
    WHITE_BALANCE_TEMPERATURE = 4900  # 参考 1231234.py white_balance_temperature_yolo = 4900
    ENCODE_FORMAT = '.jpg'       # '.jpg' 或 '.png'
    JPEG_QUALITY = 85            # JPEG质量，仅对JPEG有效

    print("=" * 50)
    print("  视频帧发送客户端 (Jetson Orin Nano)")
    print(f"  服务端: {SERVER_IP}:{SERVER_PORT}")
    print(f"  分辨率: {IMG_WIDTH}x{IMG_HEIGHT} @ {TARGET_FPS}FPS")
    print(f"  编码: {ENCODE_FORMAT} (质量={JPEG_QUALITY})")
    print(f"  曝光时间: {EXPOSURE_TIME}, 白平衡色温: {WHITE_BALANCE_TEMPERATURE}")
    print("=" * 50)

    # 1. 打开并配置摄像头
    cap = None
    try:
        cap, real_dev = setup_camera(
            cap, device_index=DEVICE_INDEX,
            img_width=IMG_WIDTH, img_height=IMG_HEIGHT, fps=TARGET_FPS,
            exposure_time=EXPOSURE_TIME,
            white_balance_temperature=WHITE_BALANCE_TEMPERATURE
        )
    except Exception as e:
        print(f"❌ 摄像头初始化失败：{e}")
        sys.exit(1)

    # 2. 连接服务端
    client = Client(ip=SERVER_IP, port=SERVER_PORT)

    # 3. 帧率控制器
    fps_controller = FrameRateController(target_fps=TARGET_FPS)

    # 4. 主循环：读取-发送
    print("\n🚀 开始发送视频帧...")
    frame_count = 0
    send_fail_count = 0
    max_consecutive_failures = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取帧失败，等待重试...")
            time.sleep(0.1)
            continue

        # 帧率控制
        fps_controller.wait()

        # 发送帧
        success = client.send_frame(frame, encode_format=ENCODE_FORMAT, quality=JPEG_QUALITY)
        if success:
            frame_count += 1
            send_fail_count = 0  # 重置连续失败计数
        else:
            send_fail_count += 1
            if send_fail_count >= max_consecutive_failures:
                print(f"❌ 连续 {max_consecutive_failures} 次发送失败，退出程序")
                break