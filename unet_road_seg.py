import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ====================== 1. 数据集 ======================
class RoadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(320, 320)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith(('jpg', 'png'))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        # 读取图片
        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0

        # 读取mask（黑白边缘图）
        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        # 转张量
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask

# ====================== 2. 自己搭建的 UNet ======================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 下采样
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)

        # 上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 输出
        self.out = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        up = self.up(d2)
        concat = torch.cat([up, d1], dim=1)
        up1 = self.up1(concat)
        return self.sigmoid(self.out(up1))

# ====================== 3. 损失函数 ======================
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def dice_loss(self, pred, target):
        smooth = 1e-8
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice_loss(pred, target)

# ====================== 4. 训练 ======================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== 你只需要改这两个路径 ==========
    img_dir = r"C:\Users\cwx\Desktop\work2\images"
    mask_dir = r"C:\Users\cwx\Desktop\work2\masks"

    # 数据
    dataset = RoadDataset(img_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 模型
    model = UNet().to(device)
    criterion = EdgeLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 开始训练
    model.train()
    print("开始训练...")
    for epoch in range(60):
        total_loss = 0
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)

            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "road_edge_model.pth")
    print("训练完成！模型已保存为 road_edge_model.pth")
    return model

# ====================== 5. 预测 ======================
def predict(model_path, test_img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 读取测试图
    img = cv2.imread(test_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 预处理
    x = cv2.resize(img, (320, 320)) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)[0, 0].cpu().numpy()

    # 二值化边缘
    edge = (pred > 0.5).astype(np.uint8) * 255
    edge = cv2.resize(edge, (w, h))

    # 显示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("原图")
    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap='gray')
    plt.title("预测边缘")
    plt.show()

    return edge

# ====================== 运行 ======================
if __name__ == "__main__":
    # 训练
    model = train()

    # 预测（换成你的测试图片路径）
    predict("road_edge_model.pth", r"C:\Users\cwx\Desktop\work2\images\test.jpg")