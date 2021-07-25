#!/usr/bin/env python3
# coding:utf-8
import cv2
import numpy as np
import time
import threading
import math
color_range=    [(34 , 37 , 126), (47 , 93 , 255)]  # 这是你想调试的hsv区间
pic_FLAG = True
camera_choice = "chest"


max_record = [0,0,0]
min_record = [255,255,255]

if pic_FLAG:
    # cap_head=cv2.VideoCapture(0)
    # cap_chest=cv2.VideoCapture(0)
    # color_mask = "green_bridge"
    stream_head = "http://192.168.43.201:8082/?action=stream?dummy=param.mjpg"
    cap_head = cv2.VideoCapture(stream_head)
    stream_chest = "http://192.168.43.201:8080/?action=stream?dummy=param.mjpg"
    cap_chest = cv2.VideoCapture(stream_chest)
else:
    # computer_v=cv2.VideoCapture(0)
    img=cv2.imread("taijie1.png")
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

def get_img():
    global ChestOrg_img, HeadOrg_img, HeadOrg_img, rawimg
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
                rawimg = np.rot90(ChestOrg)
            elif camera_choice == "head":
                rawimg = HeadOrg_img.copy()
        else:
            time.sleep(0.01)
            ret = True
            print("58L pic  error ")


# 读取图像线程
th1 = threading.Thread(target=get_img)
th1.setDaemon(True)
th1.start()

# 新建窗口
cv2.namedWindow("robotPreviewH",cv2.WINDOW_NORMAL)
cv2.namedWindow("robotPreviewH_HSV",cv2.WINDOW_NORMAL)
cv2.namedWindow("colorMask",cv2.WINDOW_NORMAL)
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

while True:

    try:

        # cv2.imshow("img2",rawimg2)
        hsvimg = cv2.cvtColor(rawimg, cv2.COLOR_BGR2HSV)
    except:
        print("hsvimg get error")
        continue
    h, s, v = cv2.split(rawimg)
    rawimg2 = hsvimg.copy()
    Imask=cv2.inRange(rawimg2,color_range[0],color_range[1])
    cv2.imshow("colorMask",Imask)
    cv2.imshow("robotPreviewH_HSV", hsvimg)
    cv2.imshow("robotPreviewH", rawimg)
    k = cv2.waitKey(500)

    # 如果按了'ESC'键，则关闭面板
    if k == 27:
        break


