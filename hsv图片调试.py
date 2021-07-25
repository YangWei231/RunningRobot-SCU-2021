#!/usr/bin/env python3
# coding:utf-8
import cv2
import numpy as np
png="taijie1.png"#本次调试使用图像

color_range={
    'blue_floor': [(100 , 185 , 155), (105 , 234 , 229)],  # yw:蓝色台阶
    'green_floor': [(69 , 155 , 86), (75 , 214 , 155)],  # yw:绿色台阶
    'red_floor1': [(0, 153 , 142), (2 , 206 , 221)],  # yw:红色台阶   我们取红色台阶需要有两个值
    'red_floor2': [(177 , 153 , 142), (179 , 206 , 221)],
    "green_bridge":[(69 , 116 , 85), (79 , 209 , 176)],
    'dilei':[(73 , 49 , 25), (107 , 130 , 59)],
}


img=cv2.imread(png)
img=np.rot90(img)
cv2.imshow("img_org",img)
hsvimg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow("hsv_img",hsvimg)
Imask_blue=cv2.inRange(hsvimg,color_range["blue_floor"][0],color_range['blue_floor'][1])
Imask_green=cv2.inRange(hsvimg,color_range["green_floor"][0],color_range['green_floor'][1])
Imask_red1=cv2.inRange(hsvimg,color_range["red_floor1"][0],color_range['red_floor1'][1])
Imask_red2=cv2.inRange(hsvimg,color_range["red_floor2"][0],color_range['red_floor2'][1])
Imask_red =cv2.bitwise_or(Imask_red1,Imask_red2)
Imask_greenbridge=cv2.inRange(hsvimg,color_range["green_bridge"][0],color_range['green_bridge'][1])
Imask_dilei=cv2.inRange(hsvimg,color_range['dilei'][0],color_range['dilei'][1])
cv2.imshow("imask_blue",Imask_blue)
cv2.imshow("imask_green",Imask_green)
cv2.imshow("imask_red",Imask_red)
cv2.imshow("imask_bridge",Imask_greenbridge)
cv2.imshow("dilei",Imask_dilei)
#####################################################
Imask_dilate=cv2.dilate(Imask_dilei,np.ones((3,3),np.uint8),iterations=2)
cv2.imshow("dilate",Imask_dilate)
cv2.waitKey()

