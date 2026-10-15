#!/bin/bash
set -e  # 错误立即退出

# ========== 1. 修复X服务器权限+DISPLAY变量 ==========
export DISPLAY=${DISPLAY:-:0}
# xhost +local:root > /dev/null 2>&1 || echo "警告：X服务器授权失败（可能无GUI环境）"

# ========== 2. 验证NVIDIA运行时（修复语法：!和sudo加空格） ==========

# ========== 3. 定义核心配置（确保变量有值，避免为空） ==========
DOCKER_IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6"  # 明确赋值，避免变量为空
# CLASH_PROXY_IP="127.0.0.1"
# CLASH_PROXY_PORT="7890"
# CAMERA_DEVICE="/dev/video0"
# SERIAL_PORT="/dev/ttyTHS1" 
# PORT1='/dev/ttyACM0'
NAME='Test1'
PIP_INSTALL_CMD="pip install flask pyserial -i https://pypi.tuna.tsinghua.edu.cn/simple"
# ========== 4. 验证镜像是否存在（修复语法：!和sudo加空格） ==========


# ========== 5. 启动Docker容器（修复续行符+参数格式） ==========
echo "启动Jetson 容器（摄像头源）..."
sudo docker run \
       -it \
       --name $NAME \
       --runtime nvidia \
       --network host \
       --privileged \
       --device-cgroup-rule='c 81:* rmw' \
       --device-cgroup-rule='c 166:* rmw' \
       -v /dev:/dev:rslave \
       -e DISPLAY=$DISPLAY \
       -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \
       -v /tmp/.X11-unix/:/tmp/.X11-unix \
       -v $(pwd)/workspace:/workspace \
       $DOCKER_IMAGE \
       /bin/bash -c "$PIP_INSTALL_CMD && /bin/bash"

# # ========== 6. 清理X权限 ==========
# xhost -local:root > /dev/null 2>&1 || true
# echo "容器已退出，恢复X服务器权限完成"
