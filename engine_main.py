from fastapi import FastAPI
import uvicorn
import cv2
import time
from ultralytics import YOLO
import torch
import time
import numpy as np
import threading
from uart_protorl import uart_protocol
import signal
import asyncio
from gpio import PIN
from OLED import OLED
import serial
import glob
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import traceback
import os
import gc
from find_video import find_video
import subprocess
from pydantic import BaseModel
from statemachine import StateMachine,State
import torch
import os, subprocess
subprocess.run(['sync'], shell=False)
with open('/proc/sys/vm/compact_memory', 'w') as f:
    f.write('1')
torch.cuda.empty_cache()
torch.cuda.synchronize()


def undistort(frame):  # 修正yolo摄像头畸变
    # 相机内参矩阵
    k = np.array([[499.43944171, 0, 329.93387913],
                 [0., 499.62830784, 210.00331522],
                 [0., 0., 1]])
    # 畸变系数
    d = np.array([[-0.4630776, 0.25775406, 0.0113158, 0.00120499, -0.04910695]])
    
    h, w = frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

def undistort_qr(frame):  # 修正二维码摄像头畸变
    # 相机内参矩阵
    k = np.array([[355.85365871, 0, 324.32839428],
                 [0, 472.35579151, 233.10696075],
                 [0, 0, 1]])
    # 畸变系数
    d = np.array([[-0.43607264, 0.29226853, -0.00341298, -0.00349869, -0.15150894]])
    
    h, w = frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

global exit_flag
exit_flag=False

log_lock=threading.Lock()

def catch_exception(exc_type, exc_value, exc_traceback):
    err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    add_log("error", f"【程序异常】\n{err_msg}")
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


class RegisterMeta(type):  #元类
    def __new__(meta, name, bases, dct):
        cls=super().__new__(meta, name, bases, dct)
        cls.register_dict={}

        for attr_name,attr in dct.items():
            if hasattr(attr,'_register_name'):
                cls.register_dict[attr._register_name]=attr_name

        return cls
    
    def __call__(cls, *args, **kwargs):
        instance=super().__call__(*args, **kwargs)
        for name,method in cls.register_dict.items():
            bound_method=getattr(instance,method)
            instance.task_request_main.register_instance_method(name,bound_method)
        return instance

class StopDetector():
    def __init__(self, thresh=1, stability_frames=7) -> None:
        self.thresh = thresh
        self.stability_frames = stability_frames
        self.prev_x = None
        self.prev_y = None
        self.consecutive_move = 0
        self.consecutive_stop = 0
        self.jump_stop = False

    def _is_stable(self, x2, y2):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x2, y2
            return False
        stable = (x2 - self.prev_x)**2 + (y2 - self.prev_y)**2 <= self.thresh**2
        # 注意：无论是否稳定，这里都更新 prev 为当前坐标（否则下一帧无法正确比较）
        self.prev_x, self.prev_y = x2, y2
        return stable

    def update(self, x, y):
        stable = self._is_stable(x, y)   # 这里已经更新了 prev

        if stable:
            # 稳定（停止）状态
            self.consecutive_stop += 1
            self.consecutive_move = 0    # 移动计数清零
        else:
            # 移动状态
            self.consecutive_move += 1
            self.consecutive_stop = 0

        print(f"consecutive_move={self.consecutive_move}, consecutive_stop={self.consecutive_stop}")

        # 触发逻辑
        if self.consecutive_stop >= self.stability_frames:
            if self.jump_stop:
                self._reset()
                return True
            else:
                self._reset()   # 第一次停止无效，仅重置计数
        elif self.consecutive_move >= self.stability_frames:
            self._reset()
            self.jump_stop = True
            print("jump_stop=True")
        return False

    def _reset(self):
        self.consecutive_move = 0
        self.consecutive_stop = 0
        # 注意：不要重置 prev_x, prev_y！否则会破坏下一帧的比较基准
        # 只重置计数器和 jump_stop 标志
        self.jump_stop = False




        
class TaskRequest():  # 注册
    def __init__(self):
        self.register_dict={}

    def register(self,name):  # 注册任务
        def wrapper(func):
            func._register_name=name
            return func
        return wrapper
    
    def get(self,name):
        return self.register_dict.get(name)  # 获取任务
    
    def register_instance_method(self,name,method):  # 注册实例方法
        self.register_dict[name]=method

class Mymachine(StateMachine):  #状态机
    def __init__(self,task_request:TaskRequest):
        self.task_request=task_request
        
        
        Mymachine.state0_m.task_request=self.task_request
        Mymachine.state1_c.task_request=self.task_request
        Mymachine.state2_m.task_request=self.task_request
        Mymachine.state3_c.task_request=self.task_request
        Mymachine.state4.task_request=self.task_request
        super().__init__()

    # ========== 父状态：转盘（复合状态 + 子状态）==========
    class state3_c(State.Compound):  # 状态3 不识别任何目标，为过度过程
        initial = True  # 默认主状态
        state3_move = State('移动',initial=True)
    class state2_m(State.Compound):  # 状态2 抓物料
        
        # 子状态（）
        state2_move = State('移动',initial=True)
        state2_grasp=State('抓取')
        # 子状态跳转（官方写法）
        to_state2_grasp=state2_move.to(state2_grasp)
        to_state2_move=state2_grasp.to(state2_move)

        def on_enter_state2_grasp(self):
            print("进入状态2_grasp，抓取")
            self.task_request.get("to2_grasp")()

        def on_enter_state2_move(self):
            print("进入状态2_move，移动")
           

        def on_exit_state2_grasp(self):
            print("退出状态2_grasp，抓取")
    class state0_m(State.Compound):  # 状态0 抓取转盘上的物料
       
        # 子状态（）

        state0_move = State('移动',initial=True)
        state0_stop = State('停止')
        state0_grasp=State('抓取')
        # 子状态跳转（官方写法）
      
      
        to_state0_stop=state0_move.to(state0_stop) | state0_grasp.to(state0_stop)
        to_state0_grasp=state0_stop.to(state0_grasp)
        to_state0_move=state0_grasp.to(state0_move)

  

        def on_enter_state0_move(self):   # 进入状态0_move，移动
            print("进入状态0_move，移动")
            add_log("info","01")
           

        # def on_enter_state0_wait(self):
        #     print("进入状态0_wait，等待")
        #     self.task_request.get("to0_wait")()

        def on_enter_state0_stop(self):
            print("进入状态0_stop，停止----------------------------------------------------------------------------")
            add_log("info","12")
            self.task_request.get("to0_stop")()

        def on_enter_state0_grasp(self):
            print("进入状态0_grasp，抓取---------------------------------------------------")
            self.task_request.get("to0_grasp")()

        def on_exit_state0_grasp(self):
            print("退出状态0_grasp，抓取")
            

    class state1_c(State.Compound):  # 状态1 放物料
        
        # 子状态（）
        state1_move = State('移动',initial=True)
        state1_grasp=State('抓取')
        # 子状态跳转（官方写法）
        to_state1_grasp=state1_move.to(state1_grasp)
        to_state1_move=state1_grasp.to(state1_move)

        def on_enter_state1_grasp(self):
            print("进入状态1_grasp，抓取")
            self.task_request.get("to1_grasp")()

        def on_enter_state1_move(self):
            print("进入状态1_move，移动")
            
        def on_exit_state1_grasp(self):
            print("退出状态1_grasp，抓取")
           


    
           


           


    class state4(State.Compound):  # 状态4 放物料
        
        
        state4_move=State('移动',initial=True)
        state4_grasp=State('抓取')

        to_state4_grasp=state4_move.to(state4_grasp)
        to_state4_move=state4_grasp.to(state4_move)

        def on_enter_state4_grasp(self):
            print("进入状态4_grasp，抓取")
            self.task_request.get("to4_grasp")()

        def on_enter_state4_move(self):
            print("进入状态4_move，移动")
            
        def on_exit_state4_grasp(self):
            print("退出状态4_grasp，抓取")
           



    to_state0=(state1_c.to(state0_m) | state2_m.to(state0_m)|state3_c.to(state0_m)|state4.to(state0_m))
    to_state1=(state0_m.to(state1_c) | state2_m.to(state1_c) | state3_c.to(state1_c) | state4.to(state1_c))
    to_state2=(state0_m.to(state2_m) | state1_c.to(state2_m) | state3_c.to(state2_m) | state4.to(state2_m))
    to_state3=(state2_m.to(state3_c) | state4.to(state3_c) | state0_m.to(state3_c) | state1_c.to(state3_c))
    to_state4=(state3_c.to(state4) | state0_m.to(state4) | state1_c.to(state4) | state2_m.to(state4))



    def on_enter_state0_m(self,*args, **kwargs):   # 进入状态0，目标转盘
        print("进入状态0，转盘")
        circle=kwargs.get("circle")
        self.task_request.get("to0")(circle)
    
    def on_enter_state1_c(self,*args, **kwargs):   # 进入状态1，目标绿圈
        print("进入状态1，绿圈")
        circle=kwargs.get("circle")
        self.task_request.get("to1")(circle)



    def on_enter_state2_m(self,*args, **kwargs):   # 进入状态2，目标绿材料
        print("进入状态2，绿材料")
        circle=kwargs.get("circle")
        self.task_request.get("to2")(circle)
    
    def on_enter_state3_c(self,*args, **kwargs):   # 进入状态3，目标绿圈
        print("进入状态3，绿圈")
        circle=kwargs.get("circle")
        self.task_request.get("to3")(circle)

    def on_enter_state4(self,*args, **kwargs):   # 进入状态4，目标绿材料
        print("进入状态4，绿材料")
        circle=kwargs.get("circle")
        self.task_request.get("to4")(circle)

   

   






class FlushPrint:  # 自定义打印类  （将打印信息同时写入html的日志）
    def __init__(self):
        pass
    
    def write(self, msg):
        if msg.strip() != "":
            add_log("info", msg)
        sys.__stdout__.write(msg)
    
    def flush(self):
        sys.__stdout__.flush()
    def isatty(self):
        return sys.__stdout__.isatty()

log_buffer=[]
def add_log(level: str, msg: str):  # 添加日志
    global log_lock
    global log_buffer
    with log_lock:
        log_buffer.append({
            "level": level,
            "time": time.strftime("%H:%M:%S"),
            "msg": msg.strip()
        })


def shut_jetson():
    os.system("sudo shutdown -h now")

def restart_jetson():
    os.system("sudo reboot")

 
def init_video_exposure(real_dev,exposure_time_yolo):  # 初始化曝光时间
    cmd_list=[
            f"v4l2-ctl -d {real_dev} -c auto_exposure=1",
            f"v4l2-ctl -d {real_dev} -c exposure_time_absolute={exposure_time_yolo}",
            f"v4l2-ctl -d {real_dev} -c backlight_compensation=0",
            f"v4l2-ctl -d {real_dev} -c exposure_dynamic_framerate=0"]
    for cmd in cmd_list:
        print(f"\n==> 执行命令: {cmd}")
        try:
            # 捕获标准输出 + 错误输出
            result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
            )
            # 命令成功
            print(f"✅ 命令执行成功")
            print(f"{result}")
            
        except subprocess.CalledProcessError as e:
            # 命令失败，会显示详细错误
            print(f"❌ 命令执行失败！")
            print(f"错误码: {e.returncode}")
            print(f"错误信息: {e.stderr.strip()}")

def init_video_white_balance(real_dev,white_balance_temperature_yolo):  # 初始化白平衡
    cmd_list=[
            f"v4l2-ctl -d {real_dev} -c white_balance_automatic=0",
            f"v4l2-ctl -d {real_dev} -c white_balance_temperature={white_balance_temperature_yolo}"]
    for cmd in cmd_list:
        print(f"\n==> 执行命令: {cmd}")
        try:
            # 捕获标准输出 + 错误输出
            result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
            )
            # 命令成功
            print(f"✅ 命令执行成功")
            
        except subprocess.CalledProcessError as e:
            # 命令失败，会显示详细错误
            print(f"❌ 命令执行失败！")
            print(f"错误码: {e.returncode}")
            print(f"错误信息: {e.stderr.strip()}")


def keep_one_per_class(results):  # 保留每个类的最多一个框
    boxes = results[0].boxes
    if len(boxes) == 0:
        return results

    best = {}
    for i, cls in enumerate(boxes.cls):
        c = int(cls)
        conf = boxes.conf[i]
        if c not in best or conf > best[c][1]:
            best[c] = [i, conf]

    keep_idx = list(best.values())
    keep_idx = [x[0] for x in keep_idx]
    results[0].boxes = results[0].boxes[keep_idx]
    return results 



class MyUart(uart_protocol):  # 自定义串口协议
    def __init__(self, port, baudrate=115200):
        super().__init__(port, baudrate)
        
    def connect(self):
        i=0
        while self.ser is None:
            if exit_flag:
                self.ser = None
                print("串口connect线程退出")
                return False
            print(f"尝试连接串口 {self.port}")

            try:
                self.ser = serial.Serial(self.port, self.baudrate,write_timeout=0)         #直接发送，不等待，防止下位机卡住，把上位机也卡住
                print("串口连接成功")
                self.pin.green_on()


            except (serial.SerialException, FileNotFoundError):
                i+=1
                print(f"无法打开串口 {self.port}")
                if i%2==1:
                    self.pin.green_off()
                else:
                    self.pin.green_on()
                acm_ports = glob.glob('/dev/ttyACM*')
                if acm_ports:
                    self.port = acm_ports[0]
                else:
                    print("未找到可用的 ACM 端口")
                    self.ser = None
                    time.sleep(1)

    

# def cv_show(frame):
#     cv2.imshow("frame",frame)
#     cv2.waitKey(1)



class VideoTackle(metaclass=RegisterMeta):
    task_request_main=TaskRequest()
    def __init__(self, camera_id='/dev/yolo_cam',camera_qr_id='/dev/qr_cam', img_width=640, img_height=480, conf_thres=0.6, model_path="/workspace/best5302.engine", interval=1,ip="127.0.0.1",port=8080,fps_debug=True,debug=True):
        self.camera_id = camera_id  # 摄像头ID
        self.img_width = img_width # 图像宽度
        self.img_height = img_height # 图像高度
        self.img_center_x=self.img_width//2 # 图像x轴中心
        self.img_center_y=self.img_height//2 # 图像y轴中心

        self.exposure_time_yolo = 360 # yolo曝光时间
        self.exposure_time_qr = 450 # qr曝光时间
        self.white_balance_temperature_yolo = 4900 # yolo白平衡温度
        self.white_balance_temperature_qr = 4900 # yolo白平衡自动模式

        self.conf_thres = conf_thres # 置信度阈值
        self.model_path = model_path # 模型路径
        
        self.ip=ip  # ip地址（flask运行地址）
        self.port=port # flask端口
        self.debug=debug # 是否开启调试
        self.fps_debug=fps_debug # 是否开启fps调试
        self.pin=PIN()
        self.pin.red_off()
        self.machine_lock = threading.Lock()
        self.idx_lock = threading.Lock()
        self.move_flag=False

        # self.oled=OLED()

        self.detector = cv2.wechat_qrcode_WeChatQRCode(  # 初始化二维码检测模型
            "qr_model/detect.prototxt",
            "qr_model/detect.caffemodel",
            "qr_model/sr.prototxt",
            "qr_model/sr.caffemodel"
        )

        self.camera_qr_id=camera_qr_id
        
        self.if_stop_flag=False # 是否停止标志位
        self.detect_flag=True # 是否进行检测标志位              --------------------------------------
        # self.yolo_thread_start=False # 是否启动yolo线程标志位

        self.min_name=''

       
        self.frame_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.last_frame = None
        self.data = {
            "r_m": [],
            "b_m": [],
            "g_m": [],
            "r_c": [],
            "b_c": [],
            "g_c": [],
            "pan": [],
        }

        if self.fps_debug:
            self.fps_count = 0
            self.fps_start_time = time.time()
            self.avg_fps = 0.0
            self.interval = interval # fps计算时间间隔
            self.flask_thread=None
        
        self.x1=None # 目标x轴左上角坐标
        self.y1=None # 目标y轴左上角坐标
        self.x2=None # 目标x轴右下角坐标
        self.y2=None # 目标y轴右下角坐标
        self.x_center=None # 目标x轴中心
        self.y_center=None # 目标y轴中心

        self.target_class={
            "r_m":[],
            "b_m":[],
            "g_m":[],
            "r_c":[],
            "b_c":[],
            "g_c":[],
            "pan":[],
        }
        
        self.exit_flag=threading.Event()


        self.cap_yolo=None  #·yolo摄像头对象
        self.cap_qr=None  #·qr摄像头对象    
        
        self.error_data={
            "r_m":b'',
            "b_m":b'',
            "g_m":b'',
            "r_c":b'',
            "b_c":b'',
            "g_c":b'',
            "pan":b'',
        }


        self.material_haved=[]  #·已处理材料/圆环列表

        self.yolo_video=None  #·yolo视频对象
        self.qr_video=None  #·qr视频对象
        self.uart=None  #·串口对象

        self.restart_flag=False # 是否重启标志位
        self.shutdown_flag=False # 是否关闭标志位
        self.rgb=[3,2,1,1,2,3]  #·rgb颜色列表

        self.wheel_time=None # 转盘运动时间计时
        self.wheel_intervel=None  #·转盘运动时间间隔
        self.idx=0  #·当前处理材料/圆环索引
        self.x_img_center_xiu=self.img_center_x+25-19  #·目标x轴中心修正
        self.y_img_center_xiu=self.img_center_y-42+11  #·目标y轴中心修正

       


        
        self.circle=1
        self.rgb_ma1=["r_m","g_m","b_m"]
        self.rgb_ma2=["r_m","g_m","b_m"]
        self.rgb_ma3=["r_m","g_m","b_m"]
        self.rgb_ca=["r_c","g_c","b_c"]
        self.flag1=False
        self.flag2=False

        self.pan_x1=50  # 取物料
        self.pan_y1=30
        self.pan_y2=-15

        self.circle_x1=1   # 圆圈放物料
        self.circle_y1=1

        self.able_x1=320-80  #判定为安全范围
        self.able_x2=320+80
        self.first_detect=True
        self.target_obj='init'

        self.state_num=0

        self.if_time=False

        self.stop_detector=StopDetector()
        self.pid_limit=70
        self.pid_time=0


    @task_request_main.register("to0")
    def init_0(self,circle):
        print(f"正在初始化0，circle={circle}")
        self.first_detect=True
        self.error_data={
            "r_m":b'',
            "b_m":b'',
            "g_m":b'',
            "pan":b'',
        }
        if circle==1 or circle==None:
            print("circle0==1")
            self.idx=0
        else:
            print("circle0==2")
            self.idx=3
        if self.if_time==True:
            self.detect_flag=False
        self.material_haved=[]
        self.move_flag=False
        print("move_flag已经置为False")
        
    @task_request_main.register("to1")
    def init_1(self,circle):
        self.flag1=False
        print(f"flag1置为false")
        print(f"正在初始化1，circle={circle}")
        
        self.error_data={
            "r_c":b'',
            "b_c":b'',
            "g_c":b'',
        }
        if circle==1:
            print("circle1==1")
            self.idx=0
        else:
            print("circle1==2")
            self.idx=3
        self.material_haved=[]
        
    @task_request_main.register("to2")
    def init_2(self,circle):
        print(f"正在初始化2，circle={circle}")
        
        self.error_data={
            "r_m":b'',
            "b_m":b'',
            "g_m":b'',
        }
        if circle==1:
            print("circle2==1")
            self.idx=0
        else:
            print("circle2==2")
            self.idx=3
        self.material_haved=[]
       

    @task_request_main.register("to3")
    def init_3(self,circle):
        print(f"正在初始化3，circle={circle}")
        
        self.first_detect=True
      
        if circle==1 or circle==None:
            print("circle0==1")
            self.idx=0
        else:
            print("circle0==2")
            self.idx=3
        if self.if_time==True:
            self.detect_flag=False
        self.material_haved=[]
        self.move_flag=False
        print("move_flag已经置为False")
        
       

        self.material_haved=[]

    @task_request_main.register("to4")
    def init_4(self,circle):
        self.flag2=False
        print(f"flag2置为false")
        print(f"正在初始化4，circle={circle}")
        
        if circle==1:
            self.idx=0
            self.error_data={
                "g_c":b'',
                "r_c":b'',
                "b_c":b'',

            }
           
            print("circle3==1")
        else:
            print("circle3==2")
            self.idx=3
            self.error_data={
                "g_c":b'',
                "r_c":b'',
                "b_c":b'',

            }
        self.material_haved=[]

   

    @task_request_main.register("to0_stop")  
    def to0_stop(self):
        print("0发送·停止指令")
        self.uart.send_data_rgb(b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00')
    
    @task_request_main.register("to0_grasp")
    def to0_grasp(self):
        print("0发送·夹取指令")
        self.uart.send_car_grap()
        _rgb_ma = self.rgb_ma1 if self.circle == 1 else (self.rgb_ma3 if self.circle == 2 else self.rgb_ma2)
        self.material_haved.append(_rgb_ma[self.rgb[self.idx]-1])
        self.idx+=1
        print('0:夹取完成，重新置位状态')
       

    

    @task_request_main.register("to1_stop")
    def to1_stop(self):
        print("1发送·停止指令")
        self.uart.send_data_rgb(b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00')


    @task_request_main.register("to1_grasp")
    def to1_grasp(self):
        print(f"grasp_time={self.grasp_time}")
        self.grasp_time=0
        self.uart.send_data_rgb(b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00')
        print("1发送·停止指令")
        time.sleep(1/30)
        print("1发送·夹取指令")
        self.uart.send_car_grap()
        _rgb_ma = self.rgb_ma1 if self.circle == 1 else (self.rgb_ma3 if self.circle == 2 else self.rgb_ma2)
        self.material_haved.append(_rgb_ma[self.rgb[self.idx]-1])
        self.idx+=1
        print('1:夹取完成，重新置位状态')
        self.pid_time=0
        

    @task_request_main.register("to2_stop")
    def to2_stop(self):
        print("2发送·停止指令")
        self.uart.send_data_rgb(b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00')

    @task_request_main.register("to2_grasp")
    def to2_grasp(self):
        print("2发送·停止指令")
        self.uart.send_data_rgb(b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00')
        time.sleep(1/30)
        print("2发送·夹取指令")
        self.uart.send_car_grap()
        _rgb_ma = self.rgb_ma1 if self.circle == 1 else (self.rgb_ma3 if self.circle == 2 else self.rgb_ma2)
        self.material_haved.append(_rgb_ma[self.rgb[self.idx]-1])
        self.idx+=1
        print('2:夹取完成，重新置位状态')
        self.pid_time=0
       

    @task_request_main.register("to3_stop")
    def to3_stop(self):
        print("3发送·停止指令")
        self.uart.send_data_rgb(b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00')

    @task_request_main.register("to3_grasp")
    def to3_grasp(self):

        
        print("3发送·停止指令")
        self.uart.send_data_rgb(b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00')
        print("3发送·夹取指令")
        self.uart.send_car_grap()
        
        print('3:夹取完成，重新置位状态')
       
            
        # self._locked_send("to_state4",circle=self.circle)
        
        # print('改为状态4')

    @task_request_main.register("to4_stop")
    def to4_stop(self):
        print("4发送·停止指令")
        self.uart.send_data_rgb(b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00')

    @task_request_main.register("to4_grasp")
    def to4_grasp(self):
        print(f"grasp_time={self.grasp_time}")
        self.grasp_time=0
        
        print("4发送·停止指令")
        self.uart.send_data_rgb(b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00'+b'\x00')
        time.sleep(1/30)
        print("4发送·夹取指令")
        self.uart.send_car_grap()
        _rgb_ma = self.rgb_ma1 if self.circle == 1 else (self.rgb_ma3 if self.circle == 2 else self.rgb_ma2)
        self.material_haved.append(_rgb_ma[self.rgb[self.idx]-1])
        self.idx+=1
        print('4:夹取完成，重新置位状态')
        self.pid_time=0
       

        

    def init_uart(self):
        self.uart=MyUart(port='/dev/ttyTHS1',baudrate=230400)
        self.uart.send_start()

    def uart_rec(self):  # 串口接收线程
        
        while True:
            if self.exit_flag.is_set():  # 退出标志位
                print("uart_rec线程退出")
                break
            
            if self.uart.ser is None:
                # self.uart.connect()
                self.restart_flag=True  # 掉串口自动重启
                continue

            elif self.uart.ser.in_waiting>0:
                print(f"当前状态: {self.machine.configuration_values[0]}")
                new_data=self.uart.ser.read()
                self.uart.receive_by_byte(new_data)

                if len(self.uart.byte_buffer)>0:
                    if self.uart.byte_buffer['cmd_id']==b'\x02' and self.uart.byte_buffer['content']==b'\x01':
                       
                        self.detect_flag=True
                        print('检测标志位开启')
      
                    
                    if self.uart.byte_buffer['cmd_id']==b'\x02' and self.uart.byte_buffer['content']==b'\x00':
                        self.detect_flag=False
                        print('检测标志位关闭')
                    if self.uart.byte_buffer['cmd_id']==b'\x08' and self.uart.byte_buffer['content']==b'\x01':
                        self.restart_flag = True
                        print('重启')

                    if self.uart.byte_buffer['cmd_id']==b'\x06' and self.uart.byte_buffer['content']==b'\x01':
                        if self.machine.configuration_values[1]=='state0_grasp':           #转盘 抓物料
                            if self.idx==3 or self.idx==6:
                                self.circle=self.circle+1
                                print(f"circle加1:{self.circle}")
                                if self.state_num==3 or self.state_num==6:
                                    self._locked_send("to_state3",circle=self.circle)
                                    
                                    print('改为状态3')
                                else:
                                    self._locked_send("to_state3",circle=self.circle)
                                    
                                    print('改为状态2')

                            else:
                                self._locked_send("to_state0_stop",circle=self.circle)
                                
                                print('改为状态0_stop')
                          


                            

                        elif self.machine.configuration_values[1]=='state2_grasp':        #圆圈拿
                            if self.idx==3 or self.idx==6:
                                self._locked_send("to_state3",circle=self.circle)
                                print('改为状态3')

                            else:
                                self._locked_send("to_state2_move",circle=self.circle)
                                
                                print('改为状态2_move')
                            
                        elif self.machine.configuration_values[0]=='state3_c':        # 状态3 不识别任何目标，为过度过程
                            self.state_num=self.state_num+1
                            if self.state_num==1:
                                time.sleep(0.8)
                                self._locked_send("to_state2",circle=self.circle)
                                print('改为状态2')
                            elif self.state_num==2:
                                self._locked_send("to_state4",circle=self.circle)
                                print('改为状态4')
                            elif self.state_num==3:
                                self._locked_send("to_state0",circle=self.circle)
                                self.detect_flag=True
                                print('改为状态0')
                            elif self.state_num==4:
                                time.sleep(0.8)
                                self._locked_send("to_state2",circle=self.circle)
                                print('改为状态2')
                            elif self.state_num==5:
                                self._locked_send("to_state4",circle=self.circle)
                                
                                print('改为状态4')
                            elif self.state_num==6:
                                self._locked_send("to_state0",circle=self.circle)
                                
                                print('改为状态0')


                            
                        elif self.machine.configuration_values[1]=='state4_grasp':        #圆圈 放
                            
                            if self.idx==3 or self.idx==6:
                                # self.circle=self.circle+1
                                # print(f"circle加1:{self.circle}")
                                self._locked_send("to_state2",circle=self.circle)
                                print('改为状态2')

                            else:
                                self._locked_send("to_state4_move",circle=self.circle)
                                
                                print('改为状态4_move')

                        print(f"当前idx: {self.idx}")




                self.uart.byte_buffer={} # 清空接收缓冲区
            time.sleep(0.001)

    def send_speed_up(self):
        direction_x=b'\x00'
        direction_y=b'\x01'
        x1_up=0
        x_up=x1_up.to_bytes(1,'little')
        y1_up=abs(60)
        y_up=y1_up.to_bytes(1,'little')
        data_s=direction_x+x_up+direction_y+y_up
        self.uart.send_data_speed(data_s)

    def send_speed_down(self):
        direction_x=b'\x00'
        direction_y=b'\x00'
        x1_up=0
        x_up=x1_up.to_bytes(1,'little')
        y1_up=abs(60)
        y_up=y1_up.to_bytes(1,'little')
        data_s=direction_x+x_up+direction_y+y_up
        self.uart.send_data_speed(data_s)

    def send_speed_right(self):
        direction_x=b'\x01'
        direction_y=b'\x00'
        x1_up=abs(60)
        x_up=x1_up.to_bytes(1,'little')
        y1_up=abs(0)
        y_up=y1_up.to_bytes(1,'little')
        data_s=direction_x+x_up+direction_y+y_up
        self.uart.send_data_speed(data_s)

    def send_speed_left(self):
        direction_x=b'\x00'
        direction_y=b'\x00'
        x1_up=abs(60)
        x_up=x1_up.to_bytes(1,'little')
        y1_up=abs(0)
        y_up=y1_up.to_bytes(1,'little')
        data_s=direction_x+x_up+direction_y+y_up
        self.uart.send_data_speed(data_s)


    def send_up(self):
        direction_x=b'\x01'
        direction_y=b'\x01'
        x1_up=0
        x_up=x1_up.to_bytes(2,'little')
        y1_up=abs(60)
        y_up=y1_up.to_bytes(2,'little')
        data_s=b'\x01'+direction_x+x_up+direction_y+y_up
        self.uart.send_data_rgb(data_s)
        print("up前进数据发送成功")

    def send_down(self):
        direction_x=b'\x01'
        direction_y=b'\x02'
        x1_down=0
        x_down=x1_down.to_bytes(2,'little')
        y1_down=abs(60)
        y_down=y1_down.to_bytes(2,'little')
        data_s=b'\x01'+direction_x+x_down+direction_y+y_down
        self.uart.send_data_rgb(data_s)
        print("down前进数据发送成功")

    def send_left(self):
        direction_x=b'\x01'
        direction_y=b'\x01'
        x1_left=abs(60)
        x_left=x1_left.to_bytes(2,'little')
        y1_left=abs(0)
        y_left=y1_left.to_bytes(2,'little')
        data_s=b'\x01'+direction_x+x_left+direction_y+y_left
        self.uart.send_data_rgb(data_s)
        print("left前进数据发送成功")

    def send_right(self):
        direction_x=b'\x02'
        direction_y=b'\x01'
        x1_right=abs(60)
        x_right=x1_right.to_bytes(2,'little')
        y1_right=abs(0)
        y_right=y1_right.to_bytes(2,'little')
        data_s=b'\x01'+direction_x+x_right+direction_y+y_right
        self.uart.send_data_rgb(data_s)
        print("right前进数据发送成功")

    def send_g(self):
        if self.machine.configuration_values[0] == 'state0_m':
            self._locked_send("to_state0_grasp")

            print("grasp0数据发送成功")
        elif self.machine.configuration_values[0] == 'state1_c':
            self._locked_send("to_state1_grasp")

            print("grasp1数据发送成功")
        elif self.machine.configuration_values[0] == 'state2_m':
            self._locked_send("to_state2_grasp")

            print("grasp2数据发送成功")
        elif self.machine.configuration_values[0] == 'state4_m':
            self._locked_send("to_state4_grasp")

            print("grasp4数据发送成功")

    def qr_detect(self):  # 二维码识别
        self.connect_camera_qr()
        
        while not self.exit_flag.is_set() and self.cap_qr.isOpened():
            ret, frame = self.cap_qr.read()
            h,w=frame.shape[:2]
            frame=undistort_qr(frame)
            
            if ret:
                result_list, points_list = self.detector.detectAndDecode(frame)

                # 如果识别到
                if len(result_list) > 0:
                    for res, pts in zip(result_list, points_list):
                        print("✅ 扫码成功：", res)

                        # 绘制框
                        cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 2)
                        cv2.putText(frame, res, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        self.cap_qr.release()
                        self.detector=None

                        
                        print("已经关闭二维码摄像头")
                        
                        # time.sleep(0.5)
                        # cv2.destroyAllWindows() 
                        self.rgb=self.uart.send_target(res)
                        print(f"接受顺序: {self.rgb}")


                self.last_frame=frame

            else:
                print("qr摄像头读取失败")
                break

        del self.cap_qr   # 释放二维码摄像头对象
        gc.collect()  # 回收内存
        self.connect_camera()  # 连接yolo摄像头
        self.yolo_thread=threading.Thread(target=self.yolo_tackle,daemon=True) # 启动yolo线程
        self.yolo_thread.start()
                
    def init_model(self):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f" 使用设备: {self.device}")
        try:
            self.model=YOLO(self.model_path,task='detect')
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def if_stop(self,x1,y1,x2,y2,thresh):
        if abs(x1-x2)**2+abs(y1-y2)**2<thresh**2:
            return True
        else:
            return False
    def connect_camera(self):
        try:
            real_dev = os.path.realpath(self.camera_id)
            real_dev=find_video(real_dev)
            self.yolo_video=real_dev
            if real_dev is None:
                print("视频设备不存在")
                return False
            self.cap_yolo=cv2.VideoCapture(real_dev,cv2.CAP_V4L2)
            self.cap_yolo.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"yolo摄像头路径: {real_dev}")
            
            print("yolo摄像头打开成功")
        except Exception as e:
            print(f"无法打开yolo摄像头: {e}")
            return False
        while not self.cap_yolo.isOpened():
            if self.exit_flag.is_set():
                print("connect_camera线程退出")
                break
            print("尝试重新打开yolo摄像头")
            time.sleep(1)
            # self.cap_yolo.open(real_dev,cv2.CAP_V4L2)
       

        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap_yolo.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_width)
        self.cap_yolo.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_height)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
        self.cap_yolo.set(cv2.CAP_PROP_FPS, 30)
        init_video_exposure(real_dev,self.exposure_time_yolo)
        init_video_white_balance(real_dev,self.white_balance_temperature_yolo)
        

        
        print("yolo摄像头设置成功")

       


    def connect_camera_qr(self):
        try:
            real_dev = os.path.realpath(self.camera_qr_id)
            print(f"dang=sdf={real_dev}")
            real_dev=find_video(real_dev)
            self.qr_video=real_dev
            if real_dev is None:
                print("视频设备不存在")
                return False
            self.cap_qr=cv2.VideoCapture(real_dev,cv2.CAP_V4L2)
            print(f"二维码摄像头路径: {real_dev}")
        except Exception as e:
            print(f"无法打开二维码摄像头: {e}")
            return False
        print("二维码摄像头打开成功")
        while not self.cap_qr.isOpened():
            if self.exit_flag.is_set():
                print("connect_camera线程退出")
                break
            print("尝试重新打开二维码摄像头")
            time.sleep(1)
            self.cap_qr.open(real_dev,cv2.CAP_V4L2)

        self.cap_qr.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap_qr.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap_qr.set(cv2.CAP_PROP_FPS, 30)
        init_video_exposure(real_dev,self.exposure_time_qr)
        init_video_white_balance(real_dev,self.white_balance_temperature_qr)

  

        print("二维码摄像头设置成功")

    def fps_calculate(self):
        self.fps_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.fps_start_time
        if elapsed_time >= self.interval:
            self.avg_fps = self.fps_count / elapsed_time
            self.fps_count = 0
            self.fps_start_time = current_time
            # print(f"平均 FPS: {self.avg_fps:.2f}")

    def _locked_send(self, event, circle=None):
        """线程安全的状态机 send，uart_rec/fastapi/yolo 共用"""
        with self.machine_lock:
            self.machine.send(event, circle=circle)

    def send_data_pid(self,name,error_x,error_y):
        print(f"name={name}")
        print(f"error_x={error_x}")
        print(f"error_y={error_y}")
        if error_x>0:
            direction_x=b'\x01'
        else:
            direction_x=b'\x02'
        error_x_abs=abs(error_x)
        error_x_s=error_x_abs.to_bytes(2,'little')
        if error_y>0:
            direction_y=b'\x01'
        else:
            direction_y=b'\x02'
        error_y_abs=abs(error_y)
        error_y_s=error_y_abs.to_bytes(2,'little')
        if name == "r_c" or name == "r_m":
            data_s=b'\x01'+direction_x+error_x_s+direction_y+error_y_s
            self.uart.send_data_rgb(data_s)
        elif name=="b_c" or name == "b_m":
            data_s=b'\x02'+direction_x+error_x_s+direction_y+error_y_s
            self.uart.send_data_rgb(data_s)
        elif name=="g_c" or name == "g_m":
            data_s=b'\x03'+direction_x+error_x_s+direction_y+error_y_s
            self.uart.send_data_rgb(data_s)

        else:
            print("name不能识别")
        
        

    def yolo_tackle(self):
        self.grasp_time=0
        self.true_stop=False
        order_m={
            'r_m':1,
            'b_m':3,
            'g_m':2,
        }
        order_c={
            'r_c':1,
            'b_c':3,
            'g_c':2,
        }
        self.data_detect={
            'state0_m':{
                "r_m":[],
                "b_m":[],
                "g_m":[],
                "pan":[],
            },
            'state1_c':{
                "r_c":[],
                "b_c":[],
                "g_c":[],
            },
            'state2_m':{
                "r_m":[],
                "b_m":[],
                "g_m":[],
            },
            'state3_c':{
                "r_c":[],
               
            },
            'state4':{
                "r_m":[],
                "b_m":[],
                "g_m":[],
                "r_c":[],
                "b_c":[],
                "g_c":[],
               
            },

  

        }
    
        consecutive_detect = 0      # 连续检测到目标的帧数
        self.target_locked = False       # 是否已锁定目标（有效）
        CONsecutive_THRESHOLD = 5   # 需要连续多少帧才认定有效
     


        
        
        frame_count=0

        self.move_flag=False   # 小车是否开始运动

        
        print("yolo_tackle启动")

        
        self.throw_num=0
        self.pan_=False
        while True:
            # print(f"frame_count:{frame_count}")
            if self.exit_flag.is_set():
                print("yolo_tackle线程退出")
                break

            ret, frame = self.cap_yolo.read()
            if self.machine.configuration_values[0] != 'state0_m':
                frame=undistort(frame)
            h, w = frame.shape[:2]
            if not ret:
                # self.cap.release()
                # self.yolo_cap = None
                print("无法读取帧")
                # self.connect_camera()
                time.sleep(0.05)  # 轻微等待即可

                continue
            # h,w=frame.shape[:2]
            # print(f"yolo:w,{w},h,{h}")   

            
            if self.debug:
                self.fps_calculate()
  
            frame_count+=1  

            if self.detect_flag and self.machine.configuration_values[0] != 'state3_c':
                with self.idx_lock:
                    idx = self.idx
                if idx >= len(self.rgb):
                    if self.circle==0:
                        idx=0
                    else:
                        idx=3
                    print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

                if self.machine.configuration_values[1]!='state0_grasp' and self.machine.configuration_values[1]!='state1_grasp' and self.machine.configuration_values[1]!='state2_grasp' and self.machine.configuration_values[1]!='state3_grasp' and self.machine.configuration_values[1]!='state4_grasp':
                    if self.machine.configuration_values[0]=='state0_m' or self.machine.configuration_values[0]=='state2_m' or (self.machine.configuration_values[0]=='state4' and self.circle ==2):
                        _rgb_ma = self.rgb_ma1 if self.circle == 1 else (self.rgb_ma3 if self.circle == 2 else self.rgb_ma2)
                        self.target_obj=_rgb_ma[self.rgb[idx]-1]
                    elif self.machine.configuration_values[0]=='state1_c' or (self.machine.configuration_values[0]=='state4' and self.circle ==1):
                        self.target_obj=self.rgb_ca[self.rgb[idx]-1]

                    elif self.machine.configuration_values[0]=='state3_c':
                        self.target_obj=self.rgb_ca[self.rgb[idx]-1]
                    print(f"目标：{self.target_obj}")
                    
                    # print("检测标志位开启,开启yolo")
                    results = self.model.predict(frame, conf=self.conf_thres, iou=0.5,device=self.device,verbose=False)
                    results=keep_one_per_class(results)
                    has_object = (results[0].boxes is not None and len(results[0].boxes) != 0)
                    if has_object:
                        consecutive_detect += 1
                        print(f"连续检测到目标{consecutive_detect}帧")
                        if consecutive_detect >= CONsecutive_THRESHOLD:
                            self.target_locked = True
                            print("锁定目标")
                       
                        # 防止误检测
                    else:
                        consecutive_detect = 0
                        self.target_locked = False
                    if  self.target_locked:  
                       
                        min_name=""   #状态0是最近的物料，其他状态指的是目标物料/圈
                        min_len=600
                        #清空字典
                        for state in self.data_detect.keys():
                            for satate1 in self.data_detect[state].keys():
                               
                                self.data_detect[state][satate1].clear()
                        self.pan_=False
                      
                        

                        result=results[0]
                        class_name=result.names
                    
                        
                        for box in result.boxes:
                            id=int(box.cls[0].cpu().numpy())
                            cls_name=class_name[id]
                            print(f"检测到的cls_name全部:{cls_name}")
                            if cls_name in self.material_haved:
                                continue
                            if cls_name not in self.error_data.keys():
                                continue
                            print(f"检测到的cls_name:{cls_name}")
                            self.x1,self.y1,self.x2,self.y2=map(int,box.xyxy[0].cpu().numpy())
                          
                            self.x_center=(self.x1+self.x2)//2

                            self.y_center=(self.y1+self.y2)//2
                            if cls_name=="pan":
                                error_x=(self.able_x1+self.able_x2)//2-self.x_center
                                error_y=150-self.y_center
                                self.pan_=True
                           

                            else:
                                if cls_name=="r_c" or cls_name=="b_c" or cls_name=="g_c" :
                                    error_x=self.x_img_center_xiu-(self.x_center-2)
                                    error_y=self.y_img_center_xiu-(self.y_center-5)
                                else:
                                    error_x=self.x_img_center_xiu-self.x_center
                                    error_y=self.y_img_center_xiu-self.y_center

                                        
                            confidence=float(box.conf[0].cpu().numpy())
                            # if cls_name in self.target_class.keys():
                            # self.target_class[cls_name]=[self.x_center,self.y_center,confidence,error_x,error_y]
                            self.data_detect[str(self.machine.configuration_values[0])][cls_name]=[self.x_center,self.y_center,confidence,error_x,error_y]
                            if cls_name in self.error_data.keys():
                                if self.machine.configuration_values[0]=='state0_m':
                                    if cls_name!="pan" :
                                       
                                        
                                        if np.sqrt(self.data_detect[str(self.machine.configuration_values[0])][cls_name][3]**2+self.data_detect[str(self.machine.configuration_values[0])][cls_name][4]**2)<min_len:
                                            min_len=np.sqrt(self.data_detect[str(self.machine.configuration_values[0])][cls_name][3]**2+self.data_detect[str(self.machine.configuration_values[0])][cls_name][4]**2)
                                            min_name=cls_name
                                        

                                else:
                                    if self.machine.configuration_values[0]=='state1_c':
                                        self.flag1=True
                                    elif self.machine.configuration_values[0]=='state4':
                                        self.flag2=True
                                    if cls_name == self.rgb_ma2[self.rgb[idx]-1] or cls_name == self.rgb_ca[self.rgb[idx]-1]:
                                        min_name=cls_name

                                       
                        


                            if self.debug:
                                
                                cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{cls_name}: {confidence:.2f}", (self.x1, self.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        print(f"idx={idx}-------------------")
                        if min_name=='' :
                           
                            print("min_name为空,虽然识别到目标,但是不是当前要识别的目标")
                            # if self.machine.configuration_values[0]=='state0_m' and len(self.data_detect['state0_m']['pan'])==0:
                            # if self.machine.configuration_values[1]=='state0_move' and self.pan_==True:
                            #     self.stop_detector._reset()
                            #     self.send_data_pid(name="r_c",error_x=self.data_detect[str(self.machine.configuration_values[0])]["pan"][3],error_y=self.data_detect[str(self.machine.configuration_values[0])]["pan"][4])

                            #     print("000000000000找pan发送成功")
                            if self.machine.configuration_values[1]=='state0_move' and self.pan_==True:
                                
                                if (abs(self.data_detect[str(self.machine.configuration_values[0])]["pan"][3])>2 and self.data_detect[str(self.machine.configuration_values[0])]["pan"][4]>2):
                                    self.send_data_pid(name="r_c",error_x=self.data_detect[str(self.machine.configuration_values[0])]["pan"][3],error_y=self.data_detect[str(self.machine.configuration_values[0])]["pan"][4])

                                    print("000000000000找pan发送成功")
                             

                            if self.machine.configuration_values[0]=='state1_c' and self.flag1:
                             

                                if idx!=0 and idx!=3:
                                    print("satet1视野中没有当前目标了")
                                    print(f"当前要识别的目标 ：   {self.rgb_ca[self.rgb[idx]-1]}，其序号：{order_c[self.rgb_ca[self.rgb[idx]-1]]}")
                                    print(f"原先要识别的目标 ：   {self.rgb_ca[self.rgb[idx-1]-1]}，其序号：{order_c[self.rgb_ca[self.rgb[idx-1]-1]]}")
                                    if  idx!=3:
                                        kx=order_c[self.rgb_ca[self.rgb[idx]-1]]-order_c[self.rgb_ca[self.rgb[idx-1]-1]]
                                        print(f"kx:{kx}")
                                        if kx<0:
                                            self.send_speed_right()
                                        else:
                                            self.send_speed_left()
                                        # error_x=kx*70
                                        # self.send_data_pid(name="r_c",error_x=error_x,error_y=0)
                                      
                                        print("补救数据1发送成功")
                                  
                                else:
                                    self.send_up()
                                    print("1这种情况应该不会出现")
                             
                              



            
                            elif self.machine.configuration_values[0]=="state2_m":

                                _rgb_ma = self.rgb_ma1 if self.circle == 1 else (self.rgb_ma3 if self.circle == 2 else self.rgb_ma2)
                                print("satet2视野中没有当前目标了")
                                print(f"当前要识别的目标 ：   {_rgb_ma[self.rgb[idx]-1]}，其序号：{order_m[_rgb_ma[self.rgb[idx]-1]]}")
                                print(f"原先要识别的目标 ：   {_rgb_ma[self.rgb[idx-1]-1]}，其序号：{order_m[_rgb_ma[self.rgb[idx-1]-1]]}")
                                if  idx!=0 and idx!=3:
                                    kx=order_m[_rgb_ma[self.rgb[idx]-1]]-order_m[_rgb_ma[self.rgb[idx-1]-1]]
                                    print(f"kx:{kx}")
                                    if kx<0:
                                        self.send_speed_right()
                                        print("2222右")
                                    else:
                                        self.send_speed_left()
                                        print("2222左")
                                    # error_x=kx*70
                                    # self.send_data_pid(name="r_m",error_x=error_x,error_y=0)

                                    print("补救数据2发送成功")

                                elif order_m[_rgb_ma[self.rgb[idx]-1]]==1:
                                    self.send_speed_right()
                                    # self.send_speed_left()
                                    print("补救数据2发送成功11")
                                else:
                                    self.send_speed_left()
                                    # self.send_speed_right()
                                    print("补救数据2发送成功22")

                                

                            elif self.machine.configuration_values[0]=="state3_c":
                                print("satet3视野中没有当前目标了")
                               
                                # self.send_data_pid(name="r_c",error_x=-70,error_y=0)
                                self.send_speed_right()
                                print("补救数据3发送成功")
                            elif self.machine.configuration_values[0]=="state4"and self.flag2:
                               

                                if idx!=0 and idx!=3:
                                    print("satet5视野中没有当前目标了")
                                    print(f"当前要识别的目标 ：   {self.rgb_ca[self.rgb[idx]-1]}，其序号：{order_c[self.rgb_ca[self.rgb[idx]-1]]}")
                                    print(f"原先要识别的目标 ：   {self.rgb_ca[self.rgb[idx-1]-1]}，其序号：{order_c[self.rgb_ca[self.rgb[idx-1]-1]]}")

                                    kx=order_c[self.rgb_ca[self.rgb[idx]-1]]-order_c[self.rgb_ca[self.rgb[idx-1]-1]]
                                    print(f"kx:{kx}")
                                    if kx<0:
                                        self.send_speed_right()
                                        print("444右")
                                    else:
                                        self.send_speed_left()
                                        print("444左")
                                    # error_x=kx*70
                                    # self.send_data_pid(name="r_c",error_x=error_x,error_y=0)
                               
                                    print("补救数据41发送成功")
                                else:
                                    
                                                                
                                  
                                    # self.send_data_pid(name="r_c",error_x=-70,error_y=0)
                                    self.send_speed_right()
                                    print("没找5补救数据42发送成功")


                            



                        if min_name in self.error_data.keys():
                            print(f"min_name:{min_name}")
                            self.min_name=min_name
                            if self.machine.configuration_values[0]=="state0_m":
                                if self.machine.configuration_values[1]=="state0_move":
                                    if self.if_time:
                                        print("进入计时模式")
                                        if self.stop_detector.update(self.data_detect[str(self.machine.configuration_values[0])][min_name][0],self.data_detect[str(self.machine.configuration_values[0])][min_name][1]):  #判断小车是否能动
                                            self.move_flag=True
                                            print("计时置为TRUE")
                                        
                                        elif self.move_flag:
                                            if  abs(self.data_detect[str(self.machine.configuration_values[0])][min_name][3])<self.pan_x1 and (self.data_detect[str(self.machine.configuration_values[0])][min_name][4]<self.pan_y1 and self.data_detect[str(self.machine.configuration_values[0])][min_name][4]>self.pan_y2): #符合条件，停下
                                                self._locked_send("to_state0_stop")
                                                print("计时模式下停下")
                                            else:
                                                self.send_data_pid(name=min_name,error_x=self.data_detect[str(self.machine.configuration_values[0])][min_name][3],error_y=self.data_detect[str(self.machine.configuration_values[0])][min_name][4])   #  继续进行PID
                                                print("计时找目标发送成功————")
                                        else:         # 小车不能运动  暂停中
                                            print("暂停中“")
                                                
                                            

                                    else:
                                        print("进入找盘模式")
                                        if len(self.data_detect[str(self.machine.configuration_values[0])]["pan"])==0:   # 未识别到 pan
    
                                            self.send_data_pid(name=min_name,error_x=self.data_detect[str(self.machine.configuration_values[0])][min_name][3],error_y=self.data_detect[str(self.machine.configuration_values[0])][min_name][4])
                                            print("01找目标发送成功————")
                                            print("未识别到盘")

                                        elif (abs(self.data_detect[str(self.machine.configuration_values[0])]["pan"][3])<2 and self.data_detect[str(self.machine.configuration_values[0])]["pan"][4]<2)  :  # 在安全范围内
                                            self._locked_send("to_state0_stop")
                                            print("0000stop")

                                        else:  # 识别到pan 没在安全范围内
                                            

                                            self.send_data_pid(name="r_c",error_x=self.data_detect[str(self.machine.configuration_values[0])]["pan"][3],error_y=self.data_detect[str(self.machine.configuration_values[0])]["pan"][4])

                                            print("01找pan发送成功")
                                        # print("进入找盘模式")
                                        # if len(self.data_detect[str(self.machine.configuration_values[0])]["pan"])==0:   # 未识别到 pan
    
                                        #     self.send_data_pid(name=min_name,error_x=self.data_detect[str(self.machine.configuration_values[0])][min_name][3],error_y=self.data_detect[str(self.machine.configuration_values[0])][min_name][4])
                                        #     print("01找目标发送成功————")
                                        #     print("未识别到盘")

                                        # elif (abs(self.data_detect[str(self.machine.configuration_values[0])]["pan"][3])<2 and self.data_detect[str(self.machine.configuration_values[0])]["pan"][4]<2)  and self.detect_m==False:  # 在安全范围内
                                        #     # self._locked_send("to_state0_stop")
                                        #     print("0000stop")
                                        #     self.detect_m=True
                                           

                                        # else:  # 识别到pan 没在安全范围内
                                        #     if self.detect_m==False:

                                        #         self.send_data_pid(name="r_c",error_x=self.data_detect[str(self.machine.configuration_values[0])]["pan"][3],error_y=self.data_detect[str(self.machine.configuration_values[0])]["pan"][4])

                                        #         print("01找pan发送成功")

                                        # if (self.able_x1)<abs(self.data_detect[str(self.machine.configuration_values[0])][min_name][0])<(self.able_x2) and (self.data_detect[str(self.machine.configuration_values[0])][min_name][1]>50) and self.detect_m:
                                        #     print("2222222222222222222222222222229")
                                        #         # if self.stop_detector.update(self.data_detect[str(self.machine.configuration_values[0])][min_name][0],self.data_detect[str(self.machine.configuration_values[0])][min_name][1]):  #判断小车是否能动
                                        #         #     self.move_flag=True
                                        #         #     print("计时置为TRUE")
                                                
                                        #         # elif self.move_flag:
                                        #     if abs(self.data_detect[str(self.machine.configuration_values[0])][min_name][3])<self.circle_x1 and abs(self.data_detect[str(self.machine.configuration_values[0])][min_name][4]<self.circle_y1 : #符合条件，停下
                                        #         self._locked_send("to_state0_stop")
                                        #         print("找盘模式下停下")
                                        #     else:
                                        #         self.send_data_pid(name=min_name,error_x=self.data_detect[str(self.machine.configuration_values[0])][min_name][3],error_y=self.data_detect[str(self.machine.configuration_values[0])][min_name][4])   #  继续进行PID
                                        #         print("找目标发送成功————") 
                                        # else:
                                        #     self.move_flag=False
                                        #     print("1111111111111111111111111111111111111111")

                                    

                                elif self.machine.configuration_values[1]=="state0_stop":
                                    if min_name==self.target_obj and (self.able_x1)<abs(self.data_detect[str(self.machine.configuration_values[0])][min_name][0])<(self.able_x2) and (self.data_detect[str(self.machine.configuration_values[0])][min_name][1]>50):
                                        if not self.first_detect:
                                            
                                            self._locked_send("to_state0_grasp")
                                            print("0检测到目标，抓取")

                                        print("第一次检测到目标物料，跳过")

                                    else:
                                        print("000000000000000000000001")
                                        self.first_detect=False
                                       
                                            

                                    
                                
                                
                            elif self.machine.configuration_values[1]=="state1_move" :
                               
                                x1_1=self.data_detect[str(self.machine.configuration_values[0])][min_name][3]
                                y1_1=self.data_detect[str(self.machine.configuration_values[0])][min_name][4]
                                if y1_1<120:
                                    if (abs(x1_1)<=self.circle_x1 and abs(y1_1)<=self.circle_y1) or self.pid_time>=self.pid_limit:
                                        
                                        self.grasp_time+=1
                                        print(f"grasp_time={self.grasp_time}")
                                        if self.grasp_time>15:
                                            self._locked_send("to_state1_grasp")
                                        
                                    else:
                                        self.grasp_time=0
                                        self.send_data_pid(name=min_name,error_x=x1_1,error_y=y1_1)
                                        print("发送状态1PID")
                                        self.pid_time+=1
                                else:
                                    

                                    self.send_data_pid(name=min_name,error_x=0,error_y=140)
                                    print("状态1前进指令发送成功")
                        
  
                                
                            elif self.machine.configuration_values[0]=="state2_m":
                                
                                x2_1=self.data_detect[str(self.machine.configuration_values[0])][min_name][3]
                                y2_1=self.data_detect[str(self.machine.configuration_values[0])][min_name][4]
                               
                                if self.machine.configuration_values[1]=="state2_move":
                                    print(f"{x2_1},{y2_1}")
                                    if (abs(x2_1)<=self.pan_x1 and abs(y2_1)<=self.pan_y1) or self.pid_time>=self.pid_limit:
                                            
                                                self._locked_send("to_state2_grasp")
                                                print("发送状态2抓取")
                                    else:
                                        
                                        self.send_data_pid(name=33,error_x=x2_1,error_y=y2_1)
                                        print("发送状态2PID")
                                        self.pid_time+=1
                            
                            elif  self.machine.configuration_values[0]=="state3_c":
                                if self.machine.configuration_values[1]=="state3_move":
                                    x3_1=self.data_detect[str(self.machine.configuration_values[0])][min_name][3]
                                    y3_1=self.data_detect[str(self.machine.configuration_values[0])][min_name][4]
                                    if abs(x3_1)<=self.pan_x1 and abs(y3_1)<=self.pan_y1:
                                        
                                
                                            self._locked_send("to_state3_grasp")
                                            print("发送状态3抓取")
                                    else:
                                        self.send_data_pid(name=min_name,error_x=x3_1,error_y=y3_1)
                                        print("发送状态3PID")

                            elif  self.machine.configuration_values[0]=="state4":
                                if self.machine.configuration_values[1]=="state4_move":
                                    x4_1=self.data_detect[str(self.machine.configuration_values[0])][min_name][3]
                                    y4_1=self.data_detect[str(self.machine.configuration_values[0])][min_name][4]
                                    if (abs(x4_1)<=self.circle_x1 and abs(y4_1)<=self.circle_y1) or self.pid_time>=self.pid_limit:
                                        self.grasp_time+=1
                                        print(f"grasp_time={self.grasp_time}")
                                        if self.grasp_time>=15:
                                
                                            self._locked_send("to_state4_grasp")
                                            print("发送状态4抓取")
                                    else:
                                        self.grasp_time=0
                                        self.send_data_pid(name=min_name,error_x=x4_1,error_y=y4_1)
                                        print("发送状态4PID")
                                        self.pid_time+=1

                            else:
                                print("未知状态")
                                        
                                       
                                        
                    else:
                        
                        if self.flag1 and self.machine.configuration_values[1]=="state1_move" and idx==0:
                            print("视野中没有目标，继续向前（可能误检一次）")
                            self.send_data_pid(name='r_c',error_x=0,error_y=140)
                            print("前进数据发送成功")

                        if self.flag2 and self.machine.configuration_values[1]=="state4_move" and idx==0:
                            print("视野中没有目标，继续向右（可能误检一次）")
                            
                            self.send_data_pid(name='r_c',error_x=-140,error_y=0)
                            print("向右数据发送成功")
                                



                        

                                
                            
                            
                                        
                                
    

                        
    

            # ====== 阈值方框可视化 ======
            if self.debug:
                # 绘制识别到的物体中心
                state_key = str(self.machine.configuration_values[0])
                if state_key in self.data_detect:
                    color_map = {
                        "r_c": (0, 0, 180),      # 深红色 (BGR)
                        "b_c": (180, 0, 0),      # 深蓝色
                        "g_c": (0, 130, 0),      # 深绿色
                        "pan": (0, 0, 0),        # 黑色
                        "r_m": (100, 100, 255),  # 浅红色
                        "b_m": (255, 100, 100),  # 浅蓝色
                        "g_m": (100, 255, 100),  # 浅绿色
                    }
                    for cls_name, data in self.data_detect[state_key].items():
                        if len(data) >= 2:
                            cx_obj, cy_obj = data[0], data[1]
                            color = color_map.get(cls_name, (255, 255, 255))
                            cv2.circle(frame, (cx_obj, cy_obj), 5, color, -1)
                            cv2.putText(frame, cls_name, (cx_obj + 8, cy_obj - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                cx = self.x_img_center_xiu
                cy = self.y_img_center_xiu
                ps = self.machine.configuration_values[0]  # 父状态
                if self.machine.configuration_values[0]=="state3_c":
                    ss=''
                else:
                    ss = self.machine.configuration_values[1]  # 子状态

                # 十字准心
                cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)
                cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (255, 255, 0), 2)
                cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (255, 255, 0), 2)

                # ---- state0：转盘 ----
                if ps == "state0_m":
                    # able 安全区 (青色竖线)
                    cv2.line(frame, (self.able_x1, 0), (self.able_x1, h), (255, 255, 0), 2)
                    cv2.line(frame, (self.able_x2, 0), (self.able_x2, h), (255, 255, 0), 2)
                    # pan 停止/抓取区 (黄色矩形，y非对称: pan_y1=10在上, pan_y2=-5在下)
                    px1 = cx - self.pan_x1; px2 = cx + self.pan_x1
                    py1 = cy - self.pan_y1; py2 = cy + abs(self.pan_y2)  # cy-10 到 cy+5
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                    cv2.putText(frame, f"stop/grasp {self.pan_x1*2}x{self.pan_y1+abs(self.pan_y2)}", (px1, py1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # ---- state1：绿圈1 ----
                elif ps == "state1_c":
                    qx1 = cx - self.circle_x1; qx2 = cx + self.circle_x1
                    qy1 = cy - self.circle_y1; qy2 = cy + self.circle_y1
                    cv2.rectangle(frame, (qx1, qy1), (qx2, qy2), (0, 255, 0), 2)
                    cv2.putText(frame, f"grasp {self.circle_x1*2}x{self.circle_y1*2}", (qx1, qy1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # ---- state2：绿材料、state3：绿圈2 ----
                elif ps in ("state2_m", "state3_c"):
                    px1 = cx - self.pan_x1; px2 = cx + self.pan_x1
                    py1 = cy - self.pan_y1; py2 = cy + self.pan_y1
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                    cv2.putText(frame, f"grasp {self.pan_x1*2}x{self.pan_y1*2}", (px1, py1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # ---- state4 ----
                elif ps == "state4":
                    qx1 = cx - self.circle_x1; qx2 = cx + self.circle_x1
                    qy1 = cy - self.circle_y1; qy2 = cy + self.circle_y1
                    cv2.rectangle(frame, (qx1, qy1), (qx2, qy2), (0, 255, 0), 2)
                    cv2.putText(frame, f"grasp {self.circle_x1*2}x{self.circle_y1*2}", (qx1, qy1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # FPS
                cv2.putText(frame, f"fps:{self.avg_fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # 当前状态标签
                cv2.putText(frame, f"{ps}/{ss}", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                with self.frame_lock:
                    self.last_frame = frame
                with self.data_lock:
                    state_key = str(self.machine.configuration_values[0])
                    state_data = self.data_detect.get(state_key, {})
                    self.data = {
                        "r_m": state_data.get("r_m", []),
                        "b_m": state_data.get("b_m", []),
                        "g_m": state_data.get("g_m", []),
                        "r_c": state_data.get("r_c", []),
                        "b_c": state_data.get("b_c", []),
                        "g_c": state_data.get("g_c", []),
                        "pan": state_data.get("pan", []),
                    }






     
    async def get_frame(self):
        loop = asyncio.get_event_loop()
        while True:
            with self.frame_lock:
                frame = self.last_frame
            if frame is not None:
                ret, jpeg = await loop.run_in_executor(
                    None, cv2.imencode, '.jpg', frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                if ret:
                    jp = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jp + b'\r\n')
                await asyncio.sleep(1/30)
            else:
                await asyncio.sleep(1/30)




    def fastapi_tackle(self):
        
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        class CameraParams(BaseModel):
            exposure_time_yolo: int
            exposure_time_qr: int
            white_balance_temperature_yolo: int
            white_balance_temperature_qr: int
            rgb: list = None
            rgb_ma1: list = None
            rgb_ma3: list = None
            # 阈值参数（可选）
            pan_x1: int = None
            pan_y1: int = None
            pan_y2: int = None
            circle_x1: int = None
            circle_y1: int = None
            able_x1: int = None
            able_x2: int = None

        class DebugParams(BaseModel):
            detect_flag: bool = None
            idx: int = None
            material_haved: list = None
            error_data: dict = None         
            parent_state: str = None         # 新：父状态
            child_state: str = None          # 新：子状态

        class StateSwitchRequest(BaseModel):  # 状态切换请求体
            event: str                       # 如 "to_state1", "to_state0_move"
            circle: int = None               # 可选，不传则用当前 self.circle

        # ---------- 状态辅助 ----------
        def get_parent_state():
            try:
                return self.machine.configuration_values[0]
            except Exception:
                return "unknown"

        def get_child_state():
            try:
                return self.machine.configuration_values[1]
            except Exception:
                return "unknown"

        # ---------- 路由 ----------
        @app.get('/video_feed')
        async def video_feed():
            return StreamingResponse(
                self.get_frame(),
                media_type='multipart/x-mixed-replace; boundary=frame'
            )

        @app.get('/target_data')
        async def target_data():
            with self.data_lock:
                result = dict(self.data)
            with self.idx_lock:
                current_idx = self.idx
            return {
                    "code": 200,
                    "result": result,
                    "parent_state": get_parent_state(),
                    "child_state": get_child_state(),
                    "min_name": self.min_name,
                    "circle": self.circle,
                    "idx": current_idx,
                    "rgb_obj_value": self.target_obj if self.target_obj else "wu",
                    "move_flag": self.move_flag,
                }

        @app.get("/state_info")
        async def state_info():
            """单独拉取当前父子状态，供前端下拉框初始化"""
            return {
                "parent_state": get_parent_state(),
                "child_state": get_child_state(),
                "circle": self.circle,
            }

        @app.post("/switch_state")
        async def switch_state(req: StateSwitchRequest):
            circle = req.circle if req.circle is not None else self.circle
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    None, self._locked_send, req.event, circle)
                return {
                    "code": 200,
                    "msg": f"事件 {req.event} 已发送",
                    "new_parent": get_parent_state(),
                    "new_child": get_child_state(),
                }
            except Exception as e:
                return {"code": 400, "msg": str(e)}

        @app.get("/shutdown")
        async def shutdown():
            self.shutdown_flag = True
            return {"code": 200, "msg": "正在关闭系统..."}

        @app.get("/restart")
        async def restart():
            self.restart_flag = True
            return {"code": 200, "msg": "重启中…"}

        @app.get("/restart_jetson")
        async def restart_jetson_api():
            restart_jetson()
            return {"code": 200, "msg": "Jetson 重启指令已发出"}

        @app.get("/shutdown_jetson")
        async def shutdown_jetson_api():
            shut_jetson()
            return {"code": 200, "msg": "Jetson 关机指令已发出"}

        @app.get("/status")
        async def get_status():
            return {
                "code": 200,
                "msg": "success",
                "detect_flag": self.detect_flag,
                "if_stop_flag": self.if_stop_flag,
                "move_flag": self.move_flag,
                "timestamp": int(time.time())
            }

        @app.post("/set_move_flag")
        async def set_move_flag(data: dict):
            val = data.get("value", False)
            self.move_flag = bool(val)
            print(f"move_flag 置为 {self.move_flag}")
            return {"code": 200, "move_flag": self.move_flag}

        @app.get("/debug_params")
        async def get_debug_params():
            error_data_str = {
                k: v.hex() if isinstance(v, bytes) else str(v)
                for k, v in self.error_data.items()
            }
            with self.idx_lock:
                current_idx = self.idx
            return {
                "code": 200,
                "detect_flag": self.detect_flag,
                "idx": current_idx,
                "material_haved": self.material_haved,
                "error_data": error_data_str,
                "parent_state": get_parent_state(),
                "child_state": get_child_state(),
            }

        @app.post("/set_debug_param")
        async def set_debug_param(params: DebugParams):
            if params.detect_flag is not None:
                self.detect_flag = params.detect_flag
                print(f"🔧 detect_flag = {self.detect_flag}")
            if params.idx is not None:
                with self.idx_lock:
                    self.idx = params.idx
                print(f"🔧 idx = {params.idx}")
            if params.material_haved is not None:
                self.material_haved = params.material_haved
                print(f"🔧 material_haved = {self.material_haved}")
            if params.error_data is not None:
                for cls_name in params.error_data:
                    if cls_name in self.error_data:
                        self.error_data[cls_name] = b''
                print(f"🔧 error_data 已重置: {list(params.error_data.keys())}")
            # parent_state / child_state 不允许从前端直接修改，忽略
            return {"code": 200, "msg": "调试参数已更新"}

        @app.get("/camera_params")
        async def camera_params():
            return {
                "exposure_time_yolo": self.exposure_time_yolo,
                "exposure_time_qr": self.exposure_time_qr,
                "white_balance_temperature_yolo": self.white_balance_temperature_yolo,
                "white_balance_temperature_qr": self.white_balance_temperature_qr,
                "x_img_center_xiu": self.x_img_center_xiu,
                "y_img_center_xiu": self.y_img_center_xiu,
                "rgb": self.rgb,
                "rgb_ma1": self.rgb_ma1,
                "rgb_ma3": self.rgb_ma3,
                "pan_x1": self.pan_x1,
                "pan_y1": self.pan_y1,
                "pan_y2": self.pan_y2,
                "circle_x1": self.circle_x1,
                "circle_y1": self.circle_y1,
                "able_x1": self.able_x1,
                "able_x2": self.able_x2,
            }

        @app.get("/get_logs")
        def get_logs():
            global log_lock, log_buffer
            with log_lock:
                logs = log_buffer.copy()
                log_buffer.clear()
            return {"logs": logs}

        @app.post("/set_camera_params")
        async def set_camera_params(params: CameraParams):
            self.exposure_time_yolo = params.exposure_time_yolo
            self.exposure_time_qr = params.exposure_time_qr
            self.white_balance_temperature_yolo = params.white_balance_temperature_yolo
            self.white_balance_temperature_qr = params.white_balance_temperature_qr
            if params.rgb is not None:
                self.rgb = params.rgb
                print("实时更新RGB数组:", self.rgb)
            if params.rgb_ma1 is not None:
                self.rgb_ma1 = params.rgb_ma1
                print("实时更新rgb_ma1数组:", self.rgb_ma1)
            if params.rgb_ma3 is not None:
                self.rgb_ma3 = params.rgb_ma3
                print("实时更新rgb_ma3数组:", self.rgb_ma3)
            # 阈值参数（实时生效，无需重启）
            if params.pan_x1 is not None: self.pan_x1 = params.pan_x1
            if params.pan_y1 is not None: self.pan_y1 = params.pan_y1
            if params.pan_y2 is not None: self.pan_y2 = params.pan_y2
            if params.circle_x1 is not None: self.circle_x1 = params.circle_x1
            if params.circle_y1 is not None: self.circle_y1 = params.circle_y1
            if params.able_x1 is not None: self.able_x1 = params.able_x1
            if params.able_x2 is not None: self.able_x2 = params.able_x2
            if self.yolo_video:
                init_video_exposure(self.yolo_video, self.exposure_time_yolo)
                init_video_white_balance(self.yolo_video, self.white_balance_temperature_yolo)
            if self.qr_video:
                init_video_exposure(self.qr_video, self.exposure_time_qr)
                init_video_white_balance(self.qr_video, self.white_balance_temperature_qr)
            return {"code": 200, "msg": "参数保存成功"}

        # ---------- 手动控制接口 ----------
        @app.get("/cmd/forward")
        async def cmd_forward():
            # self.send_up()
            self.send_speed_up()
            return {"action": "forward", "status": "ok"}

        @app.get("/cmd/back")
        async def cmd_back():
            # self.send_down()
            self.send_speed_down()
            return {"action": "back", "status": "ok"}

        @app.get("/cmd/left")
        async def cmd_left():
            # self.send_left()
            self.send_speed_left()
            # self.send_speed_left()
            return {"action": "left", "status": "ok"}

        @app.get("/cmd/right")
        async def cmd_right():
            # self.send_right()
            self.send_speed_right()
            # self.send_speed_right()
            return {"action": "right", "status": "ok"}

        @app.get("/cmd/stop")
        async def cmd_stop():
            self.send_g()
            return {"action": "stop", "status": "ok"}

        # ---------- 启动服务（优雅关闭） ----------
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=self.port,
            workers=1,
            loop="uvloop",
            reload=False,
            log_level="warning",
        )
        self.fastapi_server = uvicorn.Server(config)
        self.fastapi_server.run()
        
    def start(self):
        global exit_flag
        signal.signal(signal.SIGINT, self.release_source)
        self.machine=Mymachine(self.task_request_main)
        try:
            self.init_uart()
            if self.debug:
                self.fastapi_thread=threading.Thread(target=self.fastapi_tackle,daemon=True)
            
                self.fastapi_thread.start()
            self.uart_rec_thread=threading.Thread(target=self.uart_rec,daemon=True)
            self.uart_rec_thread.start()


            self.init_model()
            self.qr_thread=threading.Thread(target=self.qr_detect,daemon=True)
            self.qr_thread.start()
            # self.qr_thread.join()
            # self.connect_camera()
            # self.yolo_thread=threading.Thread(target=self.yolo_tackle,daemon=True)
            # self.yolo_thread.start()
        except Exception as e:
            print(f"程序初始化失败:{e}")
            self.pin.red_on()

        while not self.shutdown_flag:
            time.sleep(0.001)

            if self.restart_flag:
                self.release_source()    # 安全释放

                self.exit_flag.clear() 
                self.restart_flag=False
                
                exit_flag=False
                try:
                    time.sleep(3)
                    if self.if_time:
                        self.detect_flag=False # 是否进行检测标志位
                    self.init_uart()

                    self.uart_rec_thread=threading.Thread(target=self.uart_rec,daemon=True)
                    self.uart_rec_thread.start()


                    # self.init_model()
                    self.detector = cv2.wechat_qrcode_WeChatQRCode(
                        "qr_model/detect.prototxt",
                        "qr_model/detect.caffemodel",
                        "qr_model/sr.prototxt",
                        "qr_model/sr.caffemodel"
                    )

                    self.qr_thread=threading.Thread(target=self.qr_detect,daemon=True)
                    self.qr_thread.start()

                    # self.connect_camera()
                    # self.yolo_thread=threading.Thread(target=self.yolo_tackle,daemon=True)
                    # self.yolo_thread.start()
                except Exception as e:
                    print(f"程序初始化失败:{e}")
                    self.pin.red_on()

        self.release_source() 



    def release_source(self, signum=None, frame=None):  # 增加2个可选参数
        global exit_flag

        """资源释放方法（兼容信号处理的传参）"""
        print("开始释放资源...")
        self.pin.red_on()
        self.exit_flag.set()  # 设置退出标志
        exit_flag=True


        try:
            if hasattr(self, 'uart_rec_thread') and self.uart_rec_thread.is_alive():
                self.uart_rec_thread.join()
                print("uart_rec_thread 退出")
            if hasattr(self, 'qr_thread') and self.qr_thread.is_alive():
                self.qr_thread.join()
                print("qr_thread 退出")
            if hasattr(self, 'yolo_thread') and self.yolo_thread.is_alive():
                self.yolo_thread.join()
                print("yolo_thread 退出")
        except Exception as e:
            print(f"资源释放失败:{e}")
        time.sleep(0.5) 
        # 1. 释放视频流
        if hasattr(self, 'cap_yolo') and self.cap_yolo is not None and self.cap_yolo.isOpened():  # 先判断cap是否存在，避免空指针
            self.cap_yolo.release()
            del self.cap_yolo
                       
            self.cap_yolo = None
            gc.collect()
            os.system(f"sudo v4l2-ctl -d {self.yolo_video} --stream-mode=off > /dev/null 2>&1")
            time.sleep(0.5)
            print("yolo视频流已释放")

        if hasattr(self, 'cap_qr') and self.cap_qr is not None and self.cap_qr.isOpened():   
            self.cap_qr.release()
            del self.cap_qr
            del self.detector
            self.cap_qr = None
            gc.collect()
            print("二维码视频流已释放")
            if hasattr(self, 'detector') and self.detector is not None:
                del self.detector
                self.detector=None
                gc.collect()

        subprocess.run(['sync'], shell=False)
        with open('/proc/sys/vm/compact_memory', 'w') as f:
            f.write('1')
        self.state_num=0
        with self.idx_lock:
            self.idx=0
        print("idx==0")
        self.circle=1
        if self.machine.configuration_values[0] == 'state3_c':
            self._locked_send("to_state0",circle=self.circle)
            time.sleep(0.5)

        self._locked_send("to_state3",circle=self.circle)
        # 2. 停止Flask线程（先判断线程是否存在，再判断是否存活）
        try:
            if not self.restart_flag:
                self.shutdown_flag=True

                if hasattr(self, 'fastapi_server') and self.fastapi_server is not None:
                    self.fastapi_server.should_exit = True
                    print("已通知 FastAPI 服务器退出")
                    if hasattr(self, 'fastapi_thread') and self.fastapi_thread.is_alive():
                        self.fastapi_thread.join(timeout=3)

                if hasattr(self, 'model') and self.model is not None:
                    del self.model
                    self.model=None
                    gc.collect()
                    print("yolo模型已释放")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # 清空 PyTorch 缓存
                    print("GPU 显存已释放")
        except Exception as e:
            print(f"fastapi_thread释放失败:{e}")
        


        # 3. 关闭串口
        if hasattr(self, 'uart') and self.uart and self.uart.ser is not None:

            self.uart.ser.close()
            print("串口已关闭")
            del self.uart

            gc.collect()
            print("串口已释放")
           
        
        print("所有资源释放完成")

if __name__=="__main__":

    tackle=VideoTackle(debug=True)
    sys.stdout = FlushPrint()
    sys.excepthook = catch_exception
    # tackle.init_model()
    tackle.start()

       
            
            






