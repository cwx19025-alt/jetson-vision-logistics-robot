from __future__ import annotations
import socket
import cv2
import numpy as np
import os
import time
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import struct
from typing import Optional

# ==================== 优化后的配置参数 ====================
HOST = '0.0.0.0'
PORT = 8080
SAVE_DIR = 'saved_frames2'
RECV_BUFFER = 409600  # 优化：增大缓冲区到400KB，减少recv调用次数
FRAME_LEN_BYTES = 4
TARGET_FPS = 30  # 目标渲染帧率（控制GUI刷新频率）
FRAME_REFRESH_INTERVAL = int(1000 / TARGET_FPS)  # 刷新间隔（ms）
MAX_FRAME_QUEUE = 2  # 最大帧缓存数，避免堆积

# ==================== 全局变量（优化：加锁保护） ====================
current_frame = None
tk_image = None
is_running = True
frame_queue = []  # 帧缓存队列
frame_lock = threading.Lock()  # 帧队列锁

# ==================== 初始化准备 ====================
def init_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"创建图片保存文件夹：{os.path.abspath(SAVE_DIR)}")
    else:
        print(f"图片保存文件夹已存在：{os.path.abspath(SAVE_DIR)}")

init_save_dir()

# ==================== 保存帧函数 ====================
def save_current_frame():
    global current_frame
    with frame_lock:
        if current_frame is None:
            messagebox.showwarning("警告", "暂无有效帧可保存！")
            return
        frame_to_save = current_frame.copy()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    img_filename = f"frame_{timestamp}.jpg"
    img_save_path = os.path.join(SAVE_DIR, img_filename)

    save_success = cv2.imwrite(img_save_path, frame_to_save)
    if save_success:
        success_info = f"图片保存成功！\n路径：{os.path.abspath(img_save_path)}"
        print(success_info)
    else:
        messagebox.showerror("错误", "图片保存失败！")

# ==================== 优化：批量接收数据 ====================
def recv_all(conn: socket.socket, length: int) -> bytes | None:
    """一次性接收指定长度的数据，支持中断标志"""
    recv_data = b''
    while len(recv_data) < length and is_running:
        remaining = length - len(recv_data)
        # 优化：每次接收最大缓冲区大小，减少调用次数
        try:
            chunk = conn.recv(min(remaining, RECV_BUFFER))
        except (socket.timeout, ConnectionError):
            return None
        if not chunk:
            return None
        recv_data += chunk
    return recv_data if is_running else None

# ==================== 优化：Socket接收帧（加缓存策略、减日志） ====================
def receive_frames():
    global current_frame, is_running, frame_queue

    sk: socket.socket | None = None
    while sk is None and is_running:
        try:
            sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sk.bind((HOST, PORT))
            sk.listen(5)
            sk.settimeout(10)  # 优化：延长监听超时，减少超时触发频率
            print(f"服务端已启动，监听 {HOST}:{PORT}，等待客户端连接...")
        except Exception as e:
            messagebox.showerror("Socket错误", f"服务端初始化失败：{e}")
            is_running = False
            return

    while is_running and sk is not None:
        conn = None
        address = None
        try:
            conn, address = sk.accept()
            conn.settimeout(30)  # 优化：延长接收超时
            print(f"成功连接客户端：{address}")

            while is_running:
                # 步骤1：接收帧长度
                len_data = recv_all(conn, FRAME_LEN_BYTES)
                if len_data is None or not len_data:
                    print(f"客户端 {address} 断开连接，等待新连接...")
                    break

                # 步骤2：解析帧长度
                try:
                    frame_len = struct.unpack('>I', len_data)[0]
                    if frame_len < 10 or frame_len > 2 * 1024 * 1024:  # 放宽到2MB
                        continue
                except struct.error:
                    continue

                # 步骤3：接收帧数据
                frame_data = recv_all(conn, frame_len)
                if frame_data is None or len(frame_data) != frame_len:
                    break

                # 步骤4：解码（优化：仅必要校验，减少打印）
                try:
                    nparr = np.frombuffer(frame_data, dtype=np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                        # 优化：帧缓存队列，超过最大数则丢弃旧帧（避免延迟）
                        with frame_lock:
                            frame_queue.append(img.copy())
                            if len(frame_queue) > MAX_FRAME_QUEUE:
                                frame_queue.pop(0)  # 丢弃最旧的帧
                except Exception:
                    continue

        except socket.timeout:
            continue
        except Exception as e:
            if is_running:
                print(f"接收帧异常：{e}")
        finally:
            if conn is not None:
                conn.close()

    if sk is not None:
        sk.close()
    is_running = False
    print("Socket服务端已关闭")

# ==================== 优化：GUI刷新（降低频率、轻量化缩放） ====================
def update_gui_label(label: tk.Label):
    global current_frame, tk_image, is_running, frame_queue

    if is_running:
        # 优化：从缓存队列取最新帧，减少锁占用时间
        with frame_lock:
            if frame_queue:
                current_frame = frame_queue.pop()  # 取最新帧
                frame_queue.clear()  # 清空队列，只保留最新

        # 优化：仅当有新帧时才渲染，且使用更快的缩放算法
        if current_frame is not None:
            # 优化：使用BILINEAR插值（比LANCZOS快），减少CPU开销
            img_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize((640, 480), Image.Resampling.BILINEAR)
            tk_image = ImageTk.PhotoImage(image=img_pil)
            label.config(image=tk_image)

        # 优化：按目标FPS刷新（如30FPS则33ms刷新一次），减少主线程压力
        label.after(FRAME_REFRESH_INTERVAL, update_gui_label, label)

# ==================== 主函数 ====================
def main():
    global is_running

    root = tk.Tk()
    root.title("视频帧接收与保存工具")
    root.geometry("660x550")
    root.resizable(False, False)

    video_label = tk.Label(root, bg="black")
    video_label.pack(pady=10)

    save_btn = tk.Button(
        root,
        text="保存当前帧",
        command=save_current_frame,
        font=("微软雅黑", 12),
        width=20,
        height=2,
        bg="#4CAF50",
        fg="white"
    )
    save_btn.pack(pady=10)

    def on_window_close():
        global is_running
        is_running = False
        time.sleep(0.5)
        root.destroy()
        print("程序已退出，资源已释放")

    root.protocol("WM_DELETE_WINDOW", on_window_close)

    # 启动接收线程
    recv_thread = threading.Thread(target=receive_frames, daemon=True)
    recv_thread.start()

    update_gui_label(video_label)
    root.mainloop()

if __name__ == "__main__":
    main()