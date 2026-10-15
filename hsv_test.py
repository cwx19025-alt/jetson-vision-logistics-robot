import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, messagebox
from PIL import Image, ImageTk


class HSVAdjuster:
    def __init__(self, root):
        self.root = root
        self.root.title("HSV 调整工具")
        self.root.geometry("1200x700")

        self.image = None
        self.hsv_image = None
        self.display_image = None
        self.tk_image = None
        self.photo = None

        # HSV 偏移值
        self.h_offset = tk.IntVar(value=0)
        self.s_offset = tk.IntVar(value=0)
        self.v_offset = tk.IntVar(value=0)

        # 文件路径
        self.file_path = tk.StringVar()

        self.setup_ui()

    def setup_ui(self):
        # 顶部控制区域
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Button(control_frame, text="选择图片", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Label(control_frame, textvariable=self.file_path, fg="gray").pack(side=tk.LEFT, padx=10)

        # 主内容区域
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左侧：原图
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left_frame, text="原图", font=("Arial", 12, "bold")).pack()
        self.original_label = Label(left_frame)
        self.original_label.pack(padx=5, pady=5)

        # 右侧：HSV 调整效果
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right_frame, text="HSV 调整效果", font=("Arial", 12, "bold")).pack()
        self.hsv_label = Label(right_frame)
        self.hsv_label.pack(padx=5, pady=5)

        # 底部：HSV 控制滑块
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)

        # H 滑块
        h_frame = tk.Frame(slider_frame)
        h_frame.pack(fill=tk.X, pady=2)
        tk.Label(h_frame, text="H (色相)", width=8, anchor="w").pack(side=tk.LEFT)
        tk.Scale(
            h_frame, from_=-180, to=180, orient=tk.HORIZONTAL,
            variable=self.h_offset, command=self.update_hsv,
            length=600
        ).pack(side=tk.LEFT, padx=10)
        tk.Label(h_frame, textvariable=self.h_offset, width=4).pack(side=tk.LEFT)

        # S 滑块
        s_frame = tk.Frame(slider_frame)
        s_frame.pack(fill=tk.X, pady=2)
        tk.Label(s_frame, text="S (饱和度)", width=8, anchor="w").pack(side=tk.LEFT)
        tk.Scale(
            s_frame, from_=-255, to=255, orient=tk.HORIZONTAL,
            variable=self.s_offset, command=self.update_hsv,
            length=600
        ).pack(side=tk.LEFT, padx=10)
        tk.Label(s_frame, textvariable=self.s_offset, width=4).pack(side=tk.LEFT)

        # V 滑块
        v_frame = tk.Frame(slider_frame)
        v_frame.pack(fill=tk.X, pady=2)
        tk.Label(v_frame, text="V (明度)", width=8, anchor="w").pack(side=tk.LEFT)
        tk.Scale(
            v_frame, from_=-255, to=255, orient=tk.HORIZONTAL,
            variable=self.v_offset, command=self.update_hsv,
            length=600
        ).pack(side=tk.LEFT, padx=10)
        tk.Label(v_frame, textvariable=self.v_offset, width=4).pack(side=tk.LEFT)

        # 重置按钮
        tk.Button(slider_frame, text="重置", command=self.reset_sliders).pack(pady=5)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not path:
            return

        self.file_path.set(path.split("/")[-1].split("\\")[-1])

        # 读取图片
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            tk.messagebox.showerror("错误", "无法加载图片，请检查文件格式")
            return

        # 限定图片最大尺寸，防止显示溢出
        h, w = img_bgr.shape[:2]
        max_size = 500
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h))

        self.image = img_bgr
        self.hsv_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 显示原图
        self.display_original()
        # 显示 HSV 调整效果
        self.update_hsv()

    def display_original(self):
        if self.image is None:
            return
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self._show_image_on_label(rgb, self.original_label)

    def update_hsv(self, event=None):
        if self.hsv_image is None:
            return

        h_off = self.h_offset.get()
        s_off = self.s_offset.get()
        v_off = self.v_offset.get()

        # 复制 HSV 图像并调整
        hsv = self.hsv_image.copy().astype(np.int32)

        # H: 色相（0~179），循环处理
        hsv[:, :, 0] = (hsv[:, :, 0] + h_off) % 180

        # S: 饱和度（0~255），裁剪到有效范围
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s_off, 0, 255)

        # V: 明度（0~255），裁剪到有效范围
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + v_off, 0, 255)

        # 转回 BGR 显示
        result_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        self._show_image_on_label(result_rgb, self.hsv_label)

    def _show_image_on_label(self, rgb_array, label_widget):
        h, w = rgb_array.shape[:2]
        img = Image.fromarray(rgb_array)
        self.tk_image = ImageTk.PhotoImage(img)
        label_widget.config(image=self.tk_image)
        label_widget.image = self.tk_image

    def reset_sliders(self):
        self.h_offset.set(0)
        self.s_offset.set(0)
        self.v_offset.set(0)
        self.update_hsv()


if __name__ == "__main__":
    root = tk.Tk()
    app = HSVAdjuster(root)
    root.mainloop()