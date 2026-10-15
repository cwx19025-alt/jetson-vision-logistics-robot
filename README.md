# Jetson Orin Nano 视觉物流搬运机器人

基于 **NVIDIA Jetson Orin Nano** 的物流搬运 / 物料分拣机器人视觉与控制程序：**YOLOv8 目标检测 + 二维码任务码定位**双摄感知、**自定义串口帧协议**驱动下位机、**FastAPI 网页上位机**（MJPEG 实时监控 + 在线调参 + 手动遥控）、**python-statemachine** 状态机调度任务流程。

> 中国大学生工程实践与创新能力大赛（物流搬运赛项）参赛作品，获北京市特等奖。
> 本仓库为 **Jetson 端（视觉 / 决策 / 通信 / 上位机）** 的实现，PC 端与下位机固件不在其中。

## 功能特性

| 模块 | 说明 |
|---|---|
| 视觉感知 | 双摄方案：`/dev/yolo_cam` 做 YOLOv8 物料检测，`/dev/qr_cam` 做二维码任务码定位；640×480 采集，支持内参标定与畸变校正 |
| 模型部署 | YOLOv8 权重半精度导出 **TensorRT engine**（`export.py`，half + simplify），在 Jetson 上推理 |
| 决策调度 | `python-statemachine` 状态机驱动任务流程；装饰器注册业务接口、元类构建路由表，状态机按字符串查表调用实现业务解耦 |
| 下位机通信 | 自定义帧协议经串口 `/dev/ttyTHS1`（230400）与 STM32 交互：`2B 帧头 + 1B 指令 + 1B 长度 + payload + 1B 校验`，下发物料颜色与启停指令并解析回传 |
| 网页上位机 | FastAPI 约 20 个接口：MJPEG `/video_feed` 实时视频、相机/调试参数在线配置、`/cmd/*` 手动遥控、状态机切换、日志查看、远程重启/关机 |
| 硬件指示 | GPIO 指示灯（BOARD 29）+ OLED 屏同步显示运行状态 |
| 运行环境 | Docker（`ultralytics/ultralytics:latest-jetson-jetpack6`，`--runtime nvidia` + 设备透传 + host 网络） |

### 上位机接口（`engine_main.py`）

```
GET  /video_feed            MJPEG 实时视频流
GET  /target_data           检测 / 定位结果
GET  /state_info            状态机状态
POST /switch_state          切换状态
GET  /status                运行状态
POST /set_move_flag         运动使能
GET  /camera_params         读取相机参数      POST /set_camera_params  在线修改
GET  /debug_params          读取调试参数      POST /set_debug_param    在线修改
GET  /get_logs              运行日志
GET  /cmd/forward|back|left|right|stop       手动遥控
GET  /restart|/restart_jetson|/shutdown|/shutdown_jetson
```

## 目录结构

```
engine_main.py              主引擎：视觉 + 状态机 + 串口 + FastAPI（唯一支持三圈物料颜色表 rgb_ma1/2/3）
engine_turn_adjust.py       变体：按物料序号差 kx 判断左右转向（send_speed_right/left）
engine_colorcard_fix.py     变体：r_c/b_c/g_c 色卡误检清理 + state2_m 判定
engine_legacy.py            变体：早期单圈版（pan/able 视野判定、qr_detect、/dev/ttyACM0）
uart_protorl.py             串口帧协议库（收发/校验/按字节解析）
gpio.py                     GPIO 指示灯（LED_PIN=29，需 Jetson.GPIO）
export.py                   best.pt → TensorRT engine（half + simplify）
run.sh                      Docker 启动脚本（JetPack6 镜像 / 设备透传 / workspace 挂载）

html1/panel_monitor.html    完整监控上位机（对应主引擎全部路由）
html1/panel_debug.html      分拣调试面板
html1/rgb_ma1_config.html   物料颜色表 rgb_ma1/3 实时参数配置页（配 engine_main.py）

camera_1/                   相机标定：24 张棋盘格图 + 采集/标定脚本 + 内参结果
track/                      循迹算法实验：左边缘巡线（left_track / lite）、十字路口生成与测试图
frame_send_jetson.py        Jetson → PC 网络传帧调试（发送端）
frame_recv_pc.py            Jetson → PC 网络传帧调试（接收端）
statemachine_demo.py        状态机框架原型（python-statemachine，可独立运行）
hsv_test.py                 HSV 阈值调参 GUI
unet_road_seg.py            UNet 道路分割实验
best.pt / t2v8.pt           YOLOv8 训练权重（各约 6MB）
运维笔记.md                  板端调试记录：开机自启、串口、GPIO、v4l2 参数
```

## 快速开始（在 Jetson Orin Nano 上）

```bash
# 1) 启动容器（JetPack 6 + NVIDIA runtime，挂载 ./workspace → /workspace）
./run.sh

# 2) 容器内安装依赖（按需，镜像已含 ultralytics/torch）
pip install fastapi uvicorn pyserial -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 导出 TensorRT engine（可选，推理加速）
python export.py            # best.pt -> best.engine

# 4) 启动引擎（内嵌 FastAPI，默认 8080）
python /workspace/engine_main.py

# 5) 浏览器打开上位机
#    http://<jetson-ip>:8080/video_feed
#    或直接用 html1/panel_monitor.html（把 API_BASE 改成 Jetson 的 IP）
```

## 说明

- 工程初期存在多份同源引擎，用于适配不同现场问题；已整理为「1 主 + 3 变体」并在文件头保留对应关系（见上表）。
- 代码中的 `192.168.x.x` 均为车内局域网地址，仅作默认值示例，请按实际网络修改。
- `运维笔记.md` 为个人调试笔记，已移除其中的凭据类信息。

## 许可

个人学习与竞赛作品，仅供交流参考；如需商用请联系作者。
