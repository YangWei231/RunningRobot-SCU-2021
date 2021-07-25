""" 图片鼠标取值HSV value  camera输出显示 """
# 琚理、杨巍：该程序为取值程序。使用方式为：运行该程序后，将鼠标移到需要检测的颜色区域，
# 然后按w，这时候会取值。再按w会暂停并输出hsv的最大值和最小值。检测一个区域后按w暂停，
# 鼠标才能移到其他地方，不然鼠标移动过程中的所有颜色都会采集。
import numpy as np
import copy as cp
import cv2 
import math
import threading
import time

import matplotlib.pyplot as plt
import seaborn as sns

pic_FLAG = True
camera_choice = "chest"
photo_save="D:\juli\\"+camera_choice
# photo_path='./blackline01.jpg'
# picture_test = cv2.imread(photo_path)
# img  = picture_test.copy()


max_record = [0,0,0]
min_record = [255,255,255]


color_range = {'yellow_door': [(30, 140, 80), (40, 240, 150)],
               'black_door': [(25, 25, 10), (110, 150, 24)],
               'yellow_hole': [(25, 90, 70), (40, 255, 255)],
               'green_bridge': [(55, 60, 30), (85, 240, 175)],
               'blue_hole': [(65, 45, 40), (130, 255, 90)],
               'redball': [(0, 160, 40), (190, 255, 255)],
               'black': [(50, 30, 20), (130, 145, 50)],
               'black_hole': [(10, 10, 10), (180, 190, 60)],
               'black_gap': [(0, 0, 0), (180, 255, 80)],
               'red_floor': [(0, 55, 135), (10, 190, 190)],
               'red_floor1': [(0, 40, 115), (179, 185, 185)],
               'red_floor2': [(156, 43, 46), (180, 255, 255)],
               'chest_red_floor1': [(0, 100, 60), (20,200, 210)],
               'chest_red_floor2': [(110, 100, 60), (180,200, 210)],
               

               'Cred_floor4': [(140, 40, 45), (190, 210, 230)],
                'green_bridge': [(50, 100, 70), (70, 220, 180)],

                'black_line': [(50, 30, 20), (130, 220, 80)],
               }
#################################################初始化#########################################################
if pic_FLAG:
    # cap_head=cv2.VideoCapture(0)
    # cap_chest=cv2.VideoCapture(0)

    color_mask = "green_bridge"
    stream_head = "http://192.168.43.201:8082/?action=stream?dummy=param.mjpg"
    cap_head = cv2.VideoCapture(stream_head)
    stream_chest = "http://192.168.43.201:8080/?action=stream?dummy=param.mjpg"
    cap_chest = cv2.VideoCapture(stream_chest)
else:
    # computer_v=cv2.VideoCapture(0)
    rawimg = img.copy()


debug = True
step = 0
# state_sel = None
state_sel = 'floor'
reset = 0
skip = 0
#初始化头部舵机角度

chest_ret = False     # 读取图像标志位
ret = False     # 读取图像标志位
ChestOrg_img = None  # 原始图像更新
HeadOrg_img = None  # 原始图像更新
HeadOrg_img = None

plt_h = []
plt_s = []
plt_v = []

################################################读取图像线程#################################################
def get_img():
    global ChestOrg_img,HeadOrg_img,HeadOrg_img, rawimg
    global ret
    global cap_chest
    while True:
        if cap_chest.isOpened():
            chest_ret, ChestOrg_img = cap_chest.read()
            ret, HeadOrg_img = cap_head.read()
            if HeadOrg_img is None:
                print("HeadOrg_img error")
            if ChestOrg_img is None:
                print("ChestOrg_img error")
            if chest_ret == False:
                print("chest_ret faile")
                
            if camera_choice == "chest":
                ChestOrg = ChestOrg_img.copy()
                rawimg=np.rot90(ChestOrg)
            elif camera_choice == "head":
                rawimg = HeadOrg_img.copy()
        else:
            time.sleep(0.01)
            ret=True
            print("58L pic  error ")

# 读取图像线程
th1 = threading.Thread(target=get_img)
th1.setDaemon(True)
th1.start()





# 新建窗口
cv2.namedWindow("robotPreviewH",cv2.WINDOW_NORMAL)
cv2.namedWindow("robotPreviewH_HSV",cv2.WINDOW_NORMAL)
# cv2.namedWindow("colorMask",cv2.WINDOW_NORMAL)
if camera_choice == "head":
    cv2.resizeWindow("Camera", 640, 480)
    cv2.resizeWindow("robotPreviewH", 640, 480)
    cv2.resizeWindow("robotPreviewH_HSV", 640, 480)
    cv2.resizeWindow("colorMask", 640, 480)
elif camera_choice == "chest":
    cv2.resizeWindow("Camera", 480, 640)
    cv2.resizeWindow("robotPreviewH", 480, 640)
    cv2.resizeWindow("robotPreviewH_HSV", 480, 640)
    cv2.resizeWindow("colorMask", 480, 640)




def hsv_max(aa,bb):
    cc=[bb[0],bb[1],bb[2]]
    if aa[0]>bb[0]:
        cc[0]=aa[0]
    if aa[1]>bb[1]:
        cc[1]=aa[1]
    if aa[2]>bb[2]:
        cc[2]=aa[2]
    return cc

def hsv_min(aa,bb):
    cc=[bb[0],bb[1],bb[2]]
    if aa[0]<bb[0]:
        cc[0]=aa[0]
    if aa[1]<bb[1]:
        cc[1]=aa[1]
    if aa[2]<bb[2]:
        cc[2]=aa[2]
    return cc

def onmouse(event, x, y, flags, param):   #标准鼠标交互函数
    global max_record,min_record
    hsvimg = cv2.cvtColor(rawimg, cv2.COLOR_BGR2HSV)
    if event==cv2.EVENT_MOUSEMOVE:      #当鼠标移动时
        if sampling_flag == True:
            xy_hsv = hsvimg[y,x]
            print(x,y,xy_hsv, "正在采集，w停止采集")           #显示鼠标所在像素的数值，注意像素表示方法和坐标位置的不同
            plt_h.append(xy_hsv[0])
            plt_s.append(xy_hsv[1])
            plt_v.append(xy_hsv[2])
            max_record = hsv_max(xy_hsv,max_record)
            min_record = hsv_min(xy_hsv,min_record)

        else:
            print(min_record,"min--停止采集, w 开始采集--max",max_record)
            print("[(" + str(min_record[0]) + " , " + str(min_record[1]) + " , " + str(min_record[2]) + "), (" + str(max_record[0]) + " , " + str(max_record[1]) + " , " + str(max_record[2]) + ")]," )




# time.sleep(1)
# 创建鼠标事件的回调函数
cv2.setMouseCallback("robotPreviewH", onmouse)

num = 0
sampling_flag = False

while True:
    

    try:
        hsvimg = cv2.cvtColor(rawimg, cv2.COLOR_BGR2HSV)
    except:
        print("hsvimg get error")
        continue
    h,s,v = cv2.split(rawimg)


    
    # frame_green = cv2.inRange(hsvimg, color_range[color_mask][0],color_range[color_mask][1])  # 对原图像和掩模(颜色的字典)进行位运算
    # opened = cv2.morphologyEx(frame_green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算 去噪点
    # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算 封闭连接
    # closed = opened
    # cv2.imshow("colorMask", frame_green)


    cv2.imshow("robotPreviewH_HSV",hsvimg)
    cv2.imshow("robotPreviewH",rawimg)
    k = cv2.waitKey(500)

    # 如果按了'ESC'键，则关闭面板
    if k == 27:
        break
    if k == ord('s'):
        num += 1
        name = photo_save + str(num) + '.png'
        print(name)
        # camera_img=input()
        cv2.imwrite(name,rawimg) #保存图片

    if k ==ord('q'):
        print("请输入旧的hsv值:")
        oldrange=eval(input())
        oldmin=oldrange[0]
        oldmax=oldrange[1]
        new_max_record = hsv_max(oldmax, max_record)
        new_min_record = hsv_min(oldmin, min_record)
        print("最新的hsv值为：")
        print("[(" + str(new_min_record[0]) + " , " + str(new_min_record[1]) + " , " + str(new_min_record[2]) + "), (" + str(
            new_max_record[0]) + " , " + str(new_max_record[1]) + " , " + str(new_max_record[2]) + ")],")

        # 去掉下部分注释可以显示数据分布
            # sns.distplot(plt_h, bins = None, kde = False, hist_kws = {'color':'steelblue'}, label = 'h')
            # sns.distplot(plt_s, bins = None, kde = False, hist_kws = {'color':'purple'}, label = 's')
            # sns.distplot(plt_v, bins = None, kde = False, hist_kws = {'color':'darkgreen'}, label = 'v')
            # plt.title('hsv')
            # plt.legend()
            # plt.show()

    if k == ord('w'):
        if sampling_flag == True:
            sampling_flag = False
            print("停止采集")
        else:
            sampling_flag = True
            print("开始采集")




