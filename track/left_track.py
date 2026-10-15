import cv2
import numpy as np
import math
import time
import collections
from scipy import signal


frame=cv2.imread(r"C:\Users\cwx\Desktop\jetson orin nano\track\straight_road.png")


def draw_pts_black_white(w, h, pts, thickness=2):
    """
    创建指定宽高的纯黑色图像，将点集绘制为白色
    :param w: 图像宽度 (int)
    :param h: 图像高度 (int)
    :param pts: 点集，格式为 [[x1,y1], [x2,y2], ...] 或 [(x1,y1), (x2,y2), ...]
    :param thickness: 点的大小（圆形半径），默认2
    :return: 绘制好的黑白图像 (numpy.ndarray)
    """
    # 1. 创建 w x h 的纯黑色图像（3通道，像素值0=黑色）
    # dtype=np.uint8 是OpenCV图像的标准数据类型（0-255取值）
    black_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 2. 边界检查：如果点集为空，直接返回黑色图像
    if len(pts) == 0:
        return black_img
    
    # 3. 遍历点集，将每个点绘制为白色（255,255,255）
    white_color = (255, 255, 255)
    for pt in pts:
        # 确保点坐标是整数（OpenCV绘制需要整数坐标）
        x = int(round(pt[0]))
        y = int(round(pt[1]))
        
        # 额外防护：过滤超出图像范围的点（避免绘制报错）
        if 0 <= x < w and 0 <= y < h:
            # 绘制实心圆：图像、圆心、半径、颜色、-1=实心
            cv2.circle(black_img, (x, y), thickness, white_color, -1)
    
    return black_img

class LeftEdgeDetector:
    LINE_BLUR_KERNEL=3
    def __init__(self,frame,debug=False):
        self.stop=False
        self.debug=debug    
        self.frame=frame
        self.gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        self.dir_front=np.array([[0,-1],[1,0],[0,1],[-1,0]],dtype=np.int32) # 上右下左
        # self.dir_frontleft=np.array([[-1,-1],[1,-1],[1,1],[-1,1]]) # 不同方向下的左上
        self.dir_left=np.array([[-1,0],[0,-1],[1,0],[0,1]],dtype=np.int32) # 不同方向下的左
        self.h,self.w=self.gray.shape[:2]
        self.edge_pts_filtered=[]
        self.blur_pts=[]
        self.resampled_pts=[]
    def binary_value(self,frame,x,y,block_size):
        
        if x<0:
            return 0
        
        if y<0:
            return 0
        if x>=self.w:
            return 0
        if y>=self.h:
            return 0
        pixel=frame[int(y),int(x)]
        
        half_block_size=block_size//2
        x0=int(max(0,x-half_block_size))
        x1=int(min(self.w,x+half_block_size))
        y0=int(max(0,y-half_block_size))
        y1=int(min(self.h,y+half_block_size))
        
        block=frame[y0:y1+1,x0:x1+1]
        # return np.where(np.mean(block)<=pixel,255,0)
        if np.mean(block)<=pixel:
            return 255
        else:
            return 0
        
    def find_left_edge(self,block_size=7,start_x=None,start_y=None):
       
        if start_x is None:
            start_x=int(self.w/2)
        if start_y is None:
            start_y=int(self.h-1)
        
        assert len(self.gray.shape)==2
        assert block_size%2==1
        assert 0<=start_x<self.w
        assert 0<=start_y<self.h

        self.edge_pts=[]
        dir=0
        turn=0
        if self.binary_value(self.gray,start_x,start_y,block_size)==0:

           
            self.edge_pts.append([start_x,start_y])

        print(f'起始坐标{start_x},{start_y}')

        while(start_x>=0 and start_y>=0 and start_x<self.w and start_y<=self.h and turn<4 and not self.stop):
            if start_y==0:
                break
            x_front=start_x+self.dir_front[dir][0]
            y_front=start_y+self.dir_front[dir][1]
            # x_frontleft=start_x+self.dir_frontleft[dir][0]
            # y_frontleft=start_y+self.dir_frontleft[dir][1]
            x_left=start_x+self.dir_left[dir][0]
            y_left=start_y+self.dir_left[dir][1]
            # print({dir})
            if len(self.edge_pts)>=40 and start_x==self.edge_pts[0][0] and start_y==self.edge_pts[0][1]:
                break


                
            # elif self.binary_value(self.gray,x_frontleft,y_frontleft,block_size)==0: # 前向左侧像素为黑色 左前是墙 直走
            #     start_x=x_frontleft
            #     start_y=y_frontleft
            #     self.edge_pts.append([start_x,start_y])
            #     turn=0


            # if start_x==0 and dir==0:
            #     print(f'前方{x_front},{y_front}，{self.binary_value(self.gray,x_front,y_front,block_size)}')
            #     print(f'左侧{x_left},{y_left}，{self.binary_value(self.gray,x_left,y_left,block_size)}')
            #     # self.stop=True
            # if dir==1:
            #     self.stop=True  

            if self.binary_value(self.gray,x_left,y_left,block_size)==255: # 左侧像素为白色  左转并前进
                start_x=x_left
                start_y=y_left
                self.edge_pts.append([start_x,start_y])
                if self.debug:
                    cv2.circle(self.frame,(int(start_x),int(start_y)),1,(255,255,0),-1)

                dir=(dir-1)%4
                turn=0
                if self.debug:
                    print(f'左转并前进')


            elif self.binary_value(self.gray,x_front,y_front,block_size)==0: # 前向像素为黑色 前面是墙 右转
                dir=(dir+1)%4
                turn+=1
                if self.debug:
                    print(f'转')
    
            else:                # 前进
                start_x=x_front
                start_y=y_front
                self.edge_pts.append([start_x,start_y])
                if self.debug:
                    cv2.circle(self.frame,(int(start_x),int(start_y)),1,(255,255,0),-1)
                turn=0
                print(f'前进')

            # time.sleep(1)
            if self.debug:

                if dir==0:
                    print(f'当前方向向上')
                elif dir==1:
                    print(f'当前方向向右')
                elif dir==2:
                    print(f'当前方向向下')
                elif dir==3:
                    print(f'当前方向向左')

                print(f'当前坐标{start_x},{start_y}')

                cv2.imshow('frame',self.frame)
                cv2.waitKey(1)

        # print(f'当前方向{dir}') 

        if turn==4:
            print("未找到左边缘")


    def filter_edge_pts(self,threshld=5):
        if len(self.edge_pts)<2:
            return
        else:
            filtered_pts=[self.edge_pts[0]]
            for i in range(1,len(self.edge_pts)):
                if np.linalg.norm(np.array(self.edge_pts[i])-np.array(filtered_pts[-1]))<=threshld:
                    filtered_pts.append(self.edge_pts[i])
            self.edge_pts_filtered=filtered_pts
           
    def blur_points(self):
        if len(self.edge_pts_filtered)<self.LINE_BLUR_KERNEL:
            return
        blur_pts=[]
        half_kernel=self.LINE_BLUR_KERNEL//2
        for i in range(len(self.edge_pts_filtered)):
            x_sum=0
            y_sum=0
            cont=0
            for j in range(max(0,i-half_kernel),min(len(self.edge_pts_filtered),i+half_kernel+1)):
                x_sum+=self.edge_pts_filtered[j][0]
                y_sum+=self.edge_pts_filtered[j][1]
                cont+=1
            blur_pts.append([x_sum/cont,y_sum/cont])

       
        self.blur_pts=blur_pts

    def resample_points(self,sample_dist):
        if len(self.blur_pts)<2:
            return
        cumulative_dist=[0]
        for i in range(1,len(self.blur_pts)):
            cumulative_dist.append(cumulative_dist[-1]+np.linalg.norm(np.array(self.blur_pts[i])-np.array(self.blur_pts[i-1])))
        total_dist=cumulative_dist[-1]
        resampled_pts=[]
        current_dist=0
        idx=0
        while current_dist<=total_dist:
            while idx<len(self.blur_pts)-1 and current_dist>cumulative_dist[idx+1]:
                idx+=1
            if idx>=len(self.blur_pts)-1:
                resampled_pts.append(self.blur_pts[-1].copy())
                break
            
            t=(current_dist-cumulative_dist[idx])/(cumulative_dist[idx+1]-cumulative_dist[idx])
            x=self.blur_pts[idx][0]+t*(self.blur_pts[idx+1][0]-self.blur_pts[idx][0])
            y=self.blur_pts[idx][1]+t*(self.blur_pts[idx+1][1]-self.blur_pts[idx][1])
            resampled_pts.append([x,y])
            current_dist+=sample_dist
        self.resampled_pts=resampled_pts

    def local_angle_points(self,pts,window_size):
        pts=np.array(pts)
        angles=np.zeros(len(pts))
        for i in range(len(pts)):
            prev_idx=max(0,i-window_size//2)
            next_idx=min(len(pts)-1,i+window_size//2)
            
            vec_prev=pts[i]-pts[prev_idx]
            vec_next=pts[next_idx]-pts[i]
            dot=np.dot(vec_prev,vec_next)
            det=np.cross(vec_prev,vec_next)
            angles[i]=math.atan2(det,dot)
        return angles
    
    def nms_angle(self,angles,kernel_size):
        nms_arr=np.zeros_like(len(angles))
        half_kernel=kernel_size//2
        for i in range(len(angles)):
           start=max(0,i-half_kernel)
           end=min(len(angles)-1,i+half_kernel)
           window_angles=angles[start:end]
           max_ids=np.argmax(np.abs(window_angles))+start
           if max_ids==i:
               nms_arr[i]=angles[i]

        return nms_arr
    
    def detect_L_edge(self):
        angles=self.local_angle_points(self.blur_pts,self.LINE_BLUR_KERNEL)
        nms_arr=self.nms_angle(angles,self.LINE_BLUR_KERNEL)
        conner_idx=[]
        corner_found=False
        for i in range(len(nms_arr)):
            if nms_arr[i]==0:
                continue
            im1=max(0,i-self.LINE_BLUR_KERNEL//2)
            im2=min(len(nms_arr)-1,i+self.LINE_BLUR_KERNEL//2)
            conf=abs(abs(angles[i])-(abs(angles[im1])+abs(angles[im2]))/2)
            if 70 * math.pi / 180 < conf < 110 * math.pi / 180:
                conner_idx.append(i)
                corner_found=True
        return conner_idx,corner_found

    def fit_bottom_angle(pts,n_sample):
        if len(pts)<2:
            return 0
        pts = np.array(pts[:n_sample])
        x = pts[:, 0]
        y = pts[:, 1]

        # 最小二乘拟合直线 y = k*x + b
        k, b = np.polyfit(x, y, 1)

        # 直线斜率k = dy/dx
        # 与竖直方向夹角 = 90° - 直线与水平方向夹角
        angle_rad = math.atan2(1, abs(k))  # 竖直方向角度
        angle_deg = abs(angle_rad * 180 / math.pi - 90)

        return angle_deg












        
if __name__=="__main__":
    start_time=time.time()
    frame=cv2.imread(r"C:\Users\cwx\Desktop\jetson orin nano\track\crossroad_black_white.png")
    print(frame.shape)
    left_edge_detector=LeftEdgeDetector(frame)
    # left_edge_detector.find_left_edge()
    # # left_edge_detector.find_edge_pts()
    # left_edge_detector.filter_edge_pts()
    # left_edge_detector.blur_points()
    # left_edge_detector.resample_points(10)

    print(left_edge_detector.edge_pts)  
    draw_img=draw_pts_black_white(left_edge_detector.w,left_edge_detector.h,left_edge_detector.resampled_pts)
    cv2.imshow('draw_img',draw_img)
    cv2.waitKey(0)
    print(f'耗时{time.time()-start_time}')  
