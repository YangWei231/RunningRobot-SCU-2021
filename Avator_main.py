#!/usr/bin/env python3
# coding:utf-8

import numpy as np
import cv2
import math
import threading
import time
import datetime

import CMDcontrol

chest_r_width = 480
chest_r_height = 640
head_r_width = 640
head_r_height = 480

obs_rec = False
baf_rec = False
hole_rec = False
bridge_rec = False
door_rec = False
kick_ball_rec = False
floor_rec = False

img_debug = 0
action_DEBUG = False
box_debug = False
stream_pic = True
robot_IP = "192.168.43.201"
single_debug = 0

chest_ret = True     # 读取图像标志位
ret = False           # 读取图像标志位
ChestOrg_img = None   # 原始图像更新
HeadOrg_img = None    # 原始图像更新
ChestOrg_copy = None
HeadOrg_copy = None

sleep_time_s = 0.01
sleep_time_l = 0.05
real_test = 1       #yw:这个量为1表示是实际赛道情况，机器人会执行相应的动作，否则就只打印出现在想做什么但是并不会实际做出来。
reset = 0

if stream_pic:
    stream_head = "http://" + robot_IP +":8082/?action=stream?dummy=param.mjpg"
    cap_head = cv2.VideoCapture(stream_head)
    stream_chest = "http://" + robot_IP +":8080/?action=stream?dummy=param.mjpg"
    cap_chest = cv2.VideoCapture(stream_chest)
else:
    cap_chest = cv2.VideoCapture(0)
    cap_head = cv2.VideoCapture(2)
color_range = {
    'yellow_door':[(34 , 37 , 126), (48 , 111 , 255)],#yw:起点和终点门上的黄色
    'black_door': [(50 , 27 , 22), (96 , 156 , 46)],#yw:起点和终点门上的黑色
    'blue_baf':[(93 , 149 , 74), (105 , 252 , 152)],#yw:挡板的蓝色
    'black_dir':[(45 , 22 , 13), (128 , 135 , 57)],     #yw:地雷的黑色
    'gray_dir':[(73 , 28 , 70), (88 , 83 , 182)],#yw：地雷关卡地板的灰色
    'green_hole_chest':[(66 , 108 , 53), (76 , 243 , 168)],#yw:过坑的绿色（胸部检测）
    'green_hole_head':[(66 , 108 , 53), (76 , 243 , 168)],#yw:过坑的绿色（头部检测)
    'blue_floor':[(100 , 167 , 140), (105 , 234 , 252)],#yw:蓝色台阶
    'green_floor':[(69 , 127 , 73), (79 , 226 , 158)],#yw:绿色台阶
    'red_floor1':[(0 , 122 , 120), (3, 206 , 221)],#yw:红色台阶   我们取红色台阶需要有两个值
    'red_floor2':[(176, 122 , 120), (179 , 206 , 221)],
    'red_XP1':[(0 , 138 , 92), (3 , 212 , 186)],#yw:红色下坡   他这里取了两个掩模做了或运算  不过这两个掩模的值怎么来的我不清楚。
    'red_XP2':[(176 , 138 , 92), (179 , 212 , 186)],
    'white_ball_head': [(93 , 13 , 75), (123 , 62 , 181)],  # yw：踢的白球
    'white_ball_chest':[(0 , 0 , 89), (176 , 60 , 255)],
    'd_red_ball_floor1':[(177 , 99 , 129), (179 , 134 , 143)],#yw:这个和下面这个应该是砖
    'd_red_ball_floor2':[(75 , 18 , 182), (90 , 36 , 206)],
    'blue_hole_chest': [(111 , 86 , 111), (133 , 198 , 179)],#yw:踢球洞的蓝色圈
    'blue_hole_head' : [(112 , 90 , 49), (146 , 209 , 111)],
    'blue_hole':        [(112 , 90 , 49), (146 , 209 , 111)],
    'green_bridge':[(69 , 116 , 115), (79 , 212 , 176)],#yw:绿色桥
    'head_blue_door':[(100 , 117 , 64), (109 , 240 , 154)],#wc:蓝色门
    'green_bridge_rec': [(69 , 116 , 115), (79 , 209 , 176)],
    'blue_bridge_rec': [(102, 123, 132), (110, 213, 235)],
    'kick_ball_rec': [(4, 38, 49), (34, 132, 157)],
    #补充
    'blue_bridge':[(102 , 123 , 132), (110 , 213 , 235)],
    'green_bridge_rec':[(59 , 81 , 105), (67 , 153 , 144)],
    'blue_bridge_rec':[(102 , 123 , 132), (110 , 213 , 235)],
}

#################################################################识别
#台阶识别
def floor_detect(frame,color):#该函数输入值为图片和期待检测的颜色
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,color_range[color][0], color_range[color][1])#图像，lower，upper。在lower和upper之间的像素变为255，否则变为0
    # cv2.imshow("mask",mask)
    # cv2.waitKey(0)
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓 https://blog.csdn.net/hjxu2016/article/details/77833336/
    areaMaxContour, area_max = getAreaMaxContour1(contours)  # 找出最大轮廓
    percent = round(100 * area_max / (chest_r_width * chest_r_height), 2)  # 最大轮廓的百分比
    if areaMaxContour is not None:
        # print(percent)
        if percent>0.01:
            return 1
        else:return 0
    else:return 0

def floor_judge(frame):
    color='color11'
    if floor_detect(frame,color)==1:
        color='color22'
        if floor_detect(frame,color)==1:
            color='color33'
            if floor_detect(frame,color)==1:
                return 1
            else:return 0
        else:return 0
    else:return 0

#挡板识别
def baffle_recognize():
    global org_chest_image
    org_chest_image = ChestOrg_img.copy()
    color = 'blue_baf'
    src = org_chest_image.copy()
    src = src[int(100):int(500),int(50):int(500)]
    src = cv2.GaussianBlur(src, (5, 5), 0)
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, color_range[color][0], color_range[color][1])
    mask = cv2.dilate(mask, None, iterations=8)
    #mask = cv2.erode(mask,None,iterations=10)
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        max_area_contour,contour_max_area = getAreaMaxContour1(contours) 
        Area = cv2.contourArea(max_area_contour)
        rect = cv2.minAreaRect(max_area_contour)#最小外接矩形
        box = np.int0(cv2.boxPoints(rect))#最小外接矩形的四个顶点
        edge1=math.sqrt(math.pow(box[3, 1] - box[2, 1], 2) + math.pow(box[3, 0] - box[2, 0], 2))
        edge2=math.sqrt(math.pow(box[3, 1] - box[0, 1], 2) + math.pow(box[3, 0] - box[0, 0], 2))
        ratio=edge1/edge2   # 长与宽的比值大于3认为是条线

        # print(contour_max_area)
        # print(box)
        # print(len(contours))
        # cv2.drawContours(src,[max_area_contour],0,(0,0,255),2)
        # cv2.imshow("src",src)
        # cv2.imshow("mask",mask)
        # cv2.waitKey()
        # print(Area,ratio)

        if Area >= 5000 and ratio > 3:
            # print("正式进入挡板阶段")
            return True
        else:
            return False

#过坑识别
def hole_recognize():#yw：hole_recognize和hole_recognize_2的区别在于颜色不一样，前者是绿色，后者是蓝色。
    global org_chest_img
    org_chest_img = ChestOrg_img.copy()
    Area = 0
    color = 'green_hole_chest'
    src = org_chest_img.copy()
    src = src[int(100):int(400),int(50):int(500)]#yw：这里我记得是Y，X,切片顺序与常识不一样
    src = cv2.GaussianBlur(src, (5, 5), 0)
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, color_range[color][0], color_range[color][1])#yw：用HSV空间分割颜色有更好的效果
    closed = cv2.dilate(mask, None, iterations=5)#yw：膨胀5次，腐蚀8次。但为什么核为NONE？
    closed = cv2.erode(closed, None, iterations=8)

    # cv2.imshow("closed",closed)
    # cv2.waitKey()

    _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        max_area = max(contours, key=cv2.contourArea)
        Area = cv2.contourArea(max_area)
        rect = cv2.minAreaRect(max_area)
        #print(rect[0])
        # # print(Area)
    _, contours2, hierarchy2 = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # cv2.drawContours(src,contours2, 0, (0, 0, 255), 2)
    # print(len(contours2))
    # print(Area)
    if Area > 10000 and len(contours2) >= 2:
        return True
    else:
        return False

def hole_recognize_2():
    global org_chest_img
    org_chest_img = ChestOrg_img.copy()
    Area = 0
    color = 'blue_hole_chest'
    src = org_chest_img.copy()
    src = src[int(100):int(400),int(50):int(500)]
    src = cv2.GaussianBlur(src, (5, 5), 0)
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, color_range[color][0], color_range[color][1])
    closed = cv2.dilate(mask, None, iterations=5)
    closed = cv2.erode(closed, None, iterations=8)

    # cv2.imshow("closed",closed)
    # cv2.waitKey()

    _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        max_area = max(contours, key=cv2.contourArea)
        Area = cv2.contourArea(max_area)
        rect = cv2.minAreaRect(max_area)
        #print(rect[0])
        # # print(Area)
    _, contours2, hierarchy2 = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # cv2.drawContours(src,contours2, 0, (0, 0, 255), 2)
    # print(len(contours2))
    # print(Area)
    if Area > 20000 and len(contours2) >= 2:
        return True
    else:
        return False

#地雷识别
def tacle_recognize():
    color = 'black_dir'
    src = ChestOrg_img.copy()
    src = src[int(180):int(400),int(100):int(400)]
    src2 = HeadOrg_img.copy()
    src2 = src2[int(160):int(400),int(20):int(480)]

    # cv2.imshow("src0",src2)

    src = cv2.GaussianBlur(src, (5, 5), 0)
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, color_range[color][0], color_range[color][1])
    closed = cv2.dilate(mask, None, iterations=6)
    mask = cv2.erode(closed, None, iterations=4)
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    color2 = 'gray_dir'
    Area = 0
    src2 = cv2.GaussianBlur(src2, (5, 5), 0)
    hsv_img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_img2, color_range[color2][0], color_range[color2][1])
    mask2 = cv2.erode(mask1, None, iterations=10)
    mask3 = cv2.dilate(mask2, None, iterations=10)
    _, contours2, hierarchy2 = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # print(len(contours))
    if len(contours2) > 0: 
        max_area_contour,contour_max_area = getAreaMaxContour1(contours2) 
        Area = contour_max_area
        # print(Area)

    # cv2.imshow("mask",mask3)
    # cv2.waitKey()
    # print(len(contours))
    if Area > 15000:
        if len(contours) >= 3:
            return True
        else:
            return False

#楼梯识别
def floor_recognize():
    src = ChestOrg_img.copy()
    # src = src[int(100):int(400),int(50):int(500)]
    src = cv2.GaussianBlur(src, (5, 5), 0)
    judge = floor_judge(src)
    if judge == 1:
        return True
    else:
        return False

#过桥识别
def bridge_recognize():
    color = 'green_bridge_rec'  # 颜色变量设置为桥面所用
    contour_max_area = 0  # 初始化
    src = HeadOrg_img.copy()  # 获取头部图像的拷贝
    src = src[int(200):int(500), int(50):int(500)]  # 截取该图像的一部分内容进行处理提取轮廓
    src = cv2.GaussianBlur(src, (5, 5), 0)  # 用5*5的卷积核进行高斯模糊降噪
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)  # 将图像从rgb空间转换到hsv空间
    mask = cv2.inRange(hsv_img, color_range[color][0], color_range[color][1])  # 将符合绿色桥颜色范围的图像部分用白色显示，图像其余部分变为黑色
    mask1 = cv2.dilate(mask, None, iterations=10)  # 先进行10次膨胀
    # mask1 = cv2.erode(mask, None, iterations=10)
    mask2 = cv2.erode(mask1, None, iterations=8)  # 再进行8次腐蚀
    _, contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 用膨胀后图像去提取所有外轮廓
    # 如果存在外轮廓：
    if len(contours) > 0:
        max_area_contour, contour_max_area = getAreaMaxContour1(contours)  # 找到面积最大的轮廓以及最大轮廓的面积
        rect = cv2.minAreaRect(max_area_contour)  # 最大轮廓的最小外接矩形
        box = np.int0(cv2.boxPoints(rect))  # 找到上述最小外接矩形的四个顶点
        # 分别求出最小外接矩形的长和宽
        edge1 = math.sqrt(math.pow(box[3, 1] - box[2, 1], 2) + math.pow(box[3, 0] - box[2, 0], 2))
        edge2 = math.sqrt(math.pow(box[3, 1] - box[0, 1], 2) + math.pow(box[3, 0] - box[0, 0], 2))
        # 求出外接矩形长与宽之比
        ratio = edge1 / edge2
    # 找到经过腐蚀后图像的轮廓。建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一 个连通物体，这个物体的边界也在顶层。
    _, contours2, hierarchy2 = cv2.findContours(mask2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # cv2.imshow("mask1",mask1)
    # cv2.imshow("mask2",mask2)
    # cv2.waitKey()
    # print(Area,ratio,len(contours2))

    # if contour_max_area >= 4000 and ratio < 1.4:
    # 如果膨胀后图像的最大轮廓面积超过4000同时该最大轮廓最小外接矩形的长宽比小于1.6（意味着长宽相近或是横宽竖长，
    # 这样提取的轮廓比较接近在头部视角获得的桥的形状），并且再腐蚀后的图像轮廓也仅剩下一个，则意味着识别到了桥
    if contour_max_area >= 4000 and ratio < 1.6 and len(contours2) == 1:
        return True
    else:
        return False


# 步骤与1完全一致，只是有桥颜色上的改变
def bridge_recognize_2():
    color = 'blue_bridge_rec'
    contour_max_area = 0
    src = HeadOrg_img.copy()
    src = src[int(200):int(500), int(50):int(500)]
    src = cv2.GaussianBlur(src, (5, 5), 0)
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, color_range[color][0], color_range[color][1])
    mask1 = cv2.dilate(mask, None, iterations=10)
    # mask1 = cv2.erode(mask, None, iterations=10)
    mask2 = cv2.erode(mask1, None, iterations=8)
    _, contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        max_area_contour, contour_max_area = getAreaMaxContour1(contours)
        rect = cv2.minAreaRect(max_area_contour)  # 最小外接矩形
        box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点
        edge1 = math.sqrt(math.pow(box[3, 1] - box[2, 1], 2) + math.pow(box[3, 0] - box[2, 0], 2))
        edge2 = math.sqrt(math.pow(box[3, 1] - box[0, 1], 2) + math.pow(box[3, 0] - box[0, 0], 2))
        ratio = edge1 / edge2  # 长与宽的比值大于3认为是条线

    _, contours2, hierarchy2 = cv2.findContours(mask2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # cv2.imshow("mask1",mask1)
    # cv2.imshow("mask2",mask2)
    # cv2.waitKey()
    # print(Area,ratio,len(contours2))

    if contour_max_area >= 4000 and ratio < 1.6 and len(contours2) == 1:
        return True
    else:
        return False

#踢球识别
def kick_ball_recognize():
    color = 'kick_ball_rec'
    Area = 0
    src = HeadOrg_img.copy()
    src = cv2.GaussianBlur(src, (5, 5), 0)
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, color_range[color][0], color_range[color][1])
    # mask1 = cv2.dilate(mask, None, iterations=10)
    mask1 = cv2.erode(mask, None, iterations=10)
    mask2 = cv2.dilate(mask1, None, iterations=10)
    _, contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow("mask",mask2)
    # cv2.waitKey()
    # print(len(contours))
    if len(contours) > 0: 
        max_area_contour,contour_max_area = getAreaMaxContour1(contours) 
        Area = contour_max_area
        # print(Area)
    if Area >= 30000:
        return True
    else:
        return False

def area_calculate(color):
    contour_max_area = 0
    src = ChestOrg_img.copy()
    src = cv2.GaussianBlur(src, (5, 5), 0)
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, color_range[color][0], color_range[color][1])
    mask1 = cv2.erode(mask, None, iterations=4)
    mask2 = cv2.dilate(mask1, None, iterations=4)
    _, contours, hierarchy = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0: 
        max_area_contour,contour_max_area = getAreaMaxContour1(contours)
        return contour_max_area
    else:
        return 0

def recognize():#yw:这个函数应该是用来识别关卡的.该队伍本来准备通过这个函数识别所有关卡，但可能经过实操不大行，所以只用它识别洞和桥。
    global obs_rec
    global baf_rec
    global hole_rec
    global bridge_rec
    global door_rec
    global kick_ball_rec
    global floor_rec
    if hole_rec == False and hole_recognize():#yw：绿洞
        hole_rec = True
        return 1
    elif hole_rec == False and hole_recognize_2():#yw：蓝洞
        hole_rec = True
        return 10
    elif bridge_rec == False and bridge_recognize():#yw：绿桥
        bridge_rec = True
        return 5
    elif bridge_rec == False and bridge_recognize_2():#yw：蓝桥    不过我不明白为什么要分成一个绿的一个蓝的，是为了防止光线原因识别错误颜色吗？
        bridge_rec = True
        return 9
    # if bridge_rec == False and bridge_recognize():
    #     bridge_rec = True
    #     return 5
    # elif baf_rec == False and baffle_recognize():
    #     baf_rec = True
    #     return 3
    # elif floor_rec == False and floor_recognize():
    #     floor_rec = True
    #     return 7
    # elif kick_ball_rec == False and kick_ball_recognize():
    #     kick_ball_rec = True
    #     return 6
    # elif obs_rec == False and obstacle_recognize():
    #     obs_rec = True
    #     return 2

    else:
        return 0


acted_name = ""
def action_append(act_name):
    global acted_name

    # print("please enter to continue...")
    # cv2.waitKey(0)

    if action_DEBUG == False:
        if act_name == "forwardSlow0403" and (acted_name == "Forwalk02RL" or acted_name == "Forwalk02L"):
            acted_name = "Forwalk02LR"
        elif act_name == "forwardSlow0403" and (acted_name == "Forwalk02LR" or acted_name == "Forwalk02R"):
            acted_name = "Forwalk02RL"
        elif act_name != "forwardSlow0403" and (acted_name == "Forwalk02LR" or acted_name == "Forwalk02R"):
            # CMDcontrol.action_list.append("Forwalk02RS")
            # acted_name = act_name
            print(act_name,"动作未执行 执行 Stand")
            acted_name = "Forwalk02RS"
        elif act_name != "forwardSlow0403" and (acted_name == "Forwalk02RL" or acted_name == "Forwalk02L"):
            # CMDcontrol.action_list.append("Forwalk02LS")
            # acted_name = act_name
            print(act_name,"动作未执行 执行 Stand")
            acted_name = "Forwalk02LS"
        elif act_name == "forwardSlow0403":
            acted_name = "Forwalk02R"
        else:
            acted_name = act_name

        CMDcontrol.actionComplete = False
        if len(CMDcontrol.action_list) > 0:
            print("队列超过一个动作")
            CMDcontrol.action_list.append(acted_name)
        else:
            if single_debug:
                cv2.waitKey(0)
            CMDcontrol.action_list.append(acted_name)
        CMDcontrol.action_wait()

    else:
        print("-----------------------执行动作名：", act_name)
        time.sleep(2)


def getAreaMaxContour1(contours):    
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None
    for c in contours:  
        contour_area_temp = math.fabs(cv2.contourArea(c))  #计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > 25:  
                area_max_contour = c
    return area_max_contour, contour_area_max  


def getAreaMaxContour2(contours, area=1):
    contour_area_max = 0
    area_max_contour = None
    for c in contours:
        contour_area_temp = math.fabs(cv2.contourArea(c))
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > area:  
                area_max_contour = c
    return area_max_contour


def getLine_SumContour(contours, area=1):
    global handling
    contours_sum = None
    for c in contours:  
        area_temp = math.fabs(cv2.contourArea(c))
        rect = cv2.minAreaRect(c)  # 最小外接矩形
        box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点
        edge1 = math.sqrt(math.pow(box[3, 1] - box[0, 1], 2) + math.pow(box[3, 0] - box[0, 0], 2))
        edge2 = math.sqrt(math.pow(box[3, 1] - box[2, 1], 2) + math.pow(box[3, 0] - box[2, 0], 2))
        ratio = edge1 / edge2  # 长与宽的比值
        center_y = (box[0, 1] + box[1, 1] + box[2, 1] + box[3, 1]) / 4
        if (area_temp > area) and (ratio > 3 or ratio < 0.33) and center_y > 240:
            contours_sum = c
            break
    for c in contours:
        area_temp = math.fabs(cv2.contourArea(c))
        rect = cv2.minAreaRect(c)  # 最小外接矩形
        box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点
        edge1 = math.sqrt(math.pow(box[3, 1] - box[0, 1], 2) + math.pow(box[3, 0] - box[0, 0], 2))
        edge2 = math.sqrt(math.pow(box[3, 1] - box[2, 1], 2) + math.pow(box[3, 0] - box[2, 0], 2))
        ratio = edge1 / edge2
        # print("ratio:",ratio,"area_temp:",area_temp)

        if (area_temp > area) and (ratio > 3 or ratio < 0.33):  # 满足面积条件 长宽比条件

            rect = cv2.minAreaRect(c)  # 最小外接矩形
            box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点
            center_x = (box[0, 0] + box[1, 0] + box[2, 0] + box[3, 0]) / 4
            center_y = (box[0, 1] + box[1, 1] + box[2, 1] + box[3, 1]) / 4

            if center_y > 240:  # 满足中心点坐标条件
                contours_sum = np.concatenate((contours_sum, c), axis=0)  # 将所有轮廓点拼接到一起
                if box_debug:
                    cv2.drawContours(handling, [box], -1, (0, 255, 0), 5)
                    if img_debug:
                        cv2.imshow('handling', handling)
                        cv2.waitKey(10)
            else:
                if box_debug:
                    cv2.drawContours(handling, [box], -1, (0, 0, 255), 5)
                    if img_debug:
                        cv2.imshow('handling', handling)
                        cv2.waitKey(10)
        else:  
            rect = cv2.minAreaRect(c)  # 最小外接矩形
            box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点
            if box_debug:
                cv2.drawContours(handling, [box], -1, (0, 0, 255), 5)
                cv2.imshow('handling', handling)
                cv2.waitKey(10)

    return contours_sum

#根据颜色边缘调整角度与位置（头部）
def edge_angle(color):
    global HeadOrg_img,chest_copy, reset, skip,handling
    global handling
    angle_ok_flag = False
    angle = 90
    dis = 0
    bottom_centreX = 0
    bottom_centreY = 0
    see = False
    dis_ok_count = 0
    headTURN = 0
    hole_flag = 0

    step = 1
    while True:
        OrgFrame = HeadOrg_img.copy()
        x_start = 260
        blobs = OrgFrame[int(0):int(480), int(x_start):int(380)]  # 只对中间部分识别处理Y , X
        handling = blobs.copy()
        frame_mask = blobs.copy()

        # 获取图像中心点坐标x, y
        center = []
        #开始处理图像
        hsv = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        Imask = cv2.inRange(hsv, color_range[color][0], color_range[color][1])
        Imask = cv2.erode(Imask, None, iterations=1)
        Imask = cv2.dilate(Imask, np.ones((3, 3), np.uint8), iterations=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        Imask = cv2.morphologyEx(Imask, cv2.MORPH_OPEN, kernel)
        _, contours, hierarchy = cv2.findContours(Imask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓
        # cv2.imshow("opened",Imask)
        # print("len:",len(cnts))

        if len(contours) > 0: 
            max_area = max(contours, key=cv2.contourArea)
            epsilon = 0.05 * cv2.arcLength(max_area, True)
            approx = cv2.approxPolyDP(max_area, epsilon, True)
            approx_list = list(approx)
            approx_after = []
            for i in range(len(approx_list)):
                approx_after.append(approx_list[i][0])
            approx_sort = sorted(approx_after, key=lambda x: x[1], reverse=True)
            # if approx_sort[0][0] > approx_sort[1][0]:
            #     approx_sort[0], approx_sort[1] = approx_sort[1], approx_sort[0]
            if len(approx_sort) == 4:
                bottom_line = (approx_sort[3], approx_sort[2])
                center_x = (bottom_line[1][0]+bottom_line[0][0])/2
                center_y = (bottom_line[1][1]+bottom_line[0][1])/2
            else:
                bottom_line = None

        else:
            bottom_line = None
            
        # 初始化
        L_R_angle = 0 
        blackLine_L = [0,0]
        blackLine_R = [0,0]

        if bottom_line is not None:
            see = True
            if bottom_line[0][1] - bottom_line[1][1]==0:
                angle=90
            else:
                angle = - math.atan((bottom_line[1][1] - bottom_line[0][1]) / (bottom_line[1][0] - bottom_line[0][0]))*180.0/math.pi
            Ycenter = int((bottom_line[1][1] + bottom_line[0][1]) / 2)
            Xcenter = int((bottom_line[1][0] + bottom_line[0][0]) / 2)
            if bottom_line[1][1] > bottom_line[0][1]:
                blackLine_L = [bottom_line[1][0] , bottom_line[1][1]]
                blackLine_R = [bottom_line[0][0] , bottom_line[0][1]]
            else:
                blackLine_L =  [bottom_line[0][0] , bottom_line[0][1]]
                blackLine_R = [bottom_line[1][0] , bottom_line[1][1]]
            cv2.circle(OrgFrame, (Xcenter + x_start, Ycenter), 10, (255,255,0), -1)#画出中心点

            if blackLine_L[0] == blackLine_R[0]:
                L_R_angle = 0
            else:
                L_R_angle =  (-math.atan( (blackLine_L[1]-blackLine_R[1]) / (blackLine_L[0]-blackLine_R[0]) ) *180.0/math.pi)-4



            if img_debug:
                
                cv2.circle(OrgFrame, (blackLine_L[0] + x_start, blackLine_L[1]), 5, [0, 255, 255], 2)
                cv2.circle(OrgFrame, (blackLine_R[0] + x_start, blackLine_R[1]), 5, [255, 0, 255], 2)
                cv2.line(OrgFrame, (blackLine_R[0] + x_start,blackLine_R[1]), (blackLine_L[0] + x_start,blackLine_L[1]), (0, 255, 255), thickness=2)
                cv2.putText(OrgFrame, "L_R_angle:" + str(L_R_angle),(10, OrgFrame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(OrgFrame, "Xcenter:" + str(Xcenter + x_start),(10, OrgFrame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(OrgFrame, "Ycenter:" + str(Ycenter),(200, OrgFrame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                
                # cv2.drawContours(frame_mask, cnt_sum, -1, (255, 0, 255), 3)
                # cv2.imshow('frame_mask', frame_mask)
                cv2.imshow('black', Imask)
                cv2.imshow('OrgFrame', OrgFrame)
                cv2.waitKey(10)
        else:
            see = False
            
        #print(Ycenter)

     # 决策执行动作
        if step == 1:
            print("653L 向右看 HeadTurn015")
            action_append("HeadTurn015")
            action_append("Stand")
            time.sleep(1)   # timefftest
            step = 2

        elif step == 2:
            if not see:  # not see the edge
                # cv2.destroyAllWindows()
                print("662L 右侧看不到边缘 左侧移 Left3move")
                action_append("Left3move")
            else:   # 0
                if L_R_angle > 1.5:
                    if L_R_angle > 7:
                        headTURN += 1
                        print("668L 左大旋转 turn001L ",L_R_angle)
                        action_append("turn001L")

                    else:
                        print("672L 左旋转 turn000L ",L_R_angle)
                        headTURN += 1
                        action_append("turn000L")

                elif L_R_angle < -1.5:
                    if L_R_angle < -7:
                        headTURN += 1
                        print("679L 右大旋转  turn001R ",L_R_angle)
                        action_append("turn001R")

                    else:
                        print("683L 右旋转  turn000R ",L_R_angle)
                        action_append("turn000R")

                elif Ycenter >= 405:
                    print("687L 左侧移 Left02move > 365 ",Ycenter)
                    action_append("Left02move")

                elif Ycenter < 380:
                    print("691L 右侧移 Right02move <400 ",Ycenter)
                    action_append("Right02move")

                else:
                    print("695L 角度与位置合适 Stand")
                    action_append("Stand")
                    step = 3
                
                 
        elif step == 3:
            return 1
            break

#根据颜色边缘调整角度与位置（胸部）
def edge_angle_chest(color):
    global org_img, state, state_sel, step, reset, skip, debug   
    r_w = chest_r_width
    r_h = chest_r_height
    top_angle = 0
    T_B_angle = 0
    topcenter_x = 0.5 * r_w
    topcenter_y = 0
    bottomcenter_x = 0.5 * r_w
    bottomcenter_y = 0
    while(True):
        step = 0
        Corg_img = ChestOrg_img.copy()
        Corg_img = np.rot90(Corg_img)
        OrgFrame = Corg_img.copy()

        # 初始化 bottom_right  bottom_left
        bottom_right = (480,0)
        bottom_left =  (0,0)
        top_right = (480,0)  # 右上角点坐标
        top_left = (0,0)  # 左上角点坐标

        frame = cv2.resize(OrgFrame, (chest_r_width, chest_r_height), interpolation=cv2.INTER_LINEAR)
        frame_copy = frame.copy()
        # 获取图像中心点坐标x, y
        center = []
        # 开始处理图像
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        Imask = cv2.inRange(hsv, color_range[color][0], color_range[color][1])
        Imask = cv2.dilate(Imask, np.ones((3, 3), np.uint8), iterations=2)

        _, cnts, hierarchy = cv2.findContours(Imask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓
        
        cnt_sum, area_max = getAreaMaxContour1(cnts)  # 找出最大轮廓
        C_percent = round(area_max * 100 / (r_w * r_h), 2)  # 最大轮廓百分比
        cv2.drawContours(frame, cnt_sum, -1, (255, 0, 255), 3)

        if cnt_sum is not None:
            see = True
            rect = cv2.minAreaRect(cnt_sum)#最小外接矩形
            box = np.int0(cv2.boxPoints(rect))#最小外接矩形的四个顶点
            
            bottom_right = cnt_sum[0][0]  # 右下角点坐标
            bottom_left = cnt_sum[0][0]  # 左下角点坐标
            top_right = cnt_sum[0][0]  # 右上角点坐标
            top_left = cnt_sum[0][0]  # 左上角点坐标
            for c in cnt_sum:

                if c[0][0] + 1 * (r_h - c[0][1]) < bottom_left[0] + 1 * (r_h - bottom_left[1]):
                    bottom_left = c[0]
                if c[0][0] + 1 * c[0][1] > bottom_right[0] + 1 * bottom_right[1]:
                    bottom_right = c[0]

                if c[0][0] + 3 * c[0][1] < top_left[0] + 3 * top_left[1]:
                    top_left = c[0]
                if (r_w - c[0][0]) + 3 * c[0][1] < (r_w - top_right[0]) + 3 * top_right[1]:
                    top_right = c[0]

                # if debug:
                #     handling = ChestOrg_img.copy()
                #     cv2.circle(handling, (c[0][0], c[0][1]), 5, [0, 255, 0], 2)
                #     cv2.circle(handling, (bottom_left[0], bottom_left[1]), 5, [255, 255, 0], 2)
                #     cv2.circle(handling, (bottom_right[0], bottom_right[1]), 5, [255, 0, 255], 2)
                #     cv2.imshow('handling', handling)  # 显示图像
                #     cv2.waitKey(2)

            bottomcenter_x = (bottom_left[0] + bottom_right[0]) / 2  # 得到bottom中心坐标
            bottomcenter_y = (bottom_left[1] + bottom_right[1]) / 2

            topcenter_x = (top_right[0] + top_left[0]) / 2  # 得到top中心坐标
            topcenter_y = (top_left[1] + top_right[1]) / 2

            bottom_angle =  -math.atan( (bottom_right[1]-bottom_left[1]) / (bottom_right[0]-bottom_left[0]) ) *180.0/math.pi
            top_angle =  -math.atan( (top_right[1]-top_left[1]) / (top_right[0]-top_left[0]) ) *180.0/math.pi
            if math.fabs(topcenter_x - bottomcenter_x) <= 1:  # 得到连线的角度
                T_B_angle = 90
            else:
                T_B_angle = - math.atan((topcenter_y - bottomcenter_y) / (topcenter_x - bottomcenter_x)) * 180.0 / math.pi

            if img_debug:
                cv2.drawContours(frame_copy, [box], 0, (0, 255, 0), 2)  # 将大矩形画在图上
                cv2.line(frame_copy, (bottom_left[0],bottom_left[1]), (bottom_right[0],bottom_right[1]), (255, 255, 0), thickness=2)
                cv2.line(frame_copy, (top_left[0],top_left[1]), (top_right[0],top_right[1]), (255, 255, 0), thickness=2)
                cv2.line(frame_copy, (int(bottomcenter_x),int(bottomcenter_y)), (int(topcenter_x),int(topcenter_y)), (255, 255, 255), thickness=2)    # T_B_line

                cv2.putText(frame_copy, "bottom_angle:" + str(bottom_angle), (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),2)  # (0, 0, 255)BGR
                cv2.putText(frame_copy, "top_angle:" + str(top_angle),(30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                cv2.putText(frame_copy, "T_B_angle:" + str(T_B_angle),(30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                cv2.putText(frame_copy, "bottomcenter_x:" + str(bottomcenter_x), (30, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),2)  # (0, 0, 255)BGR
                cv2.putText(frame_copy, "y:" + str(int(bottomcenter_y)), (300, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0, 0, 0), 2)  # (0, 0, 255)BGR

                cv2.putText(frame_copy, "topcenter_x:" + str(topcenter_x), (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),2)  # (0, 0, 255)BGR
                cv2.putText(frame_copy, "topcenter_y:" + str(int(topcenter_y)), (230, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0, 0, 0), 2)  # (0, 0, 255)BGR

                cv2.putText(frame_copy, 'C_percent:' + str(C_percent) + '%', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                cv2.putText(frame_copy, "step:" + str(step), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),2)  # (0, 0, 255)BGR
                
                cv2.circle(frame_copy, (int(topcenter_x), int(topcenter_y)), 5, [255, 0, 255], 2)
                cv2.circle(frame_copy, (int(bottomcenter_x), int(bottomcenter_y)), 5, [255, 0, 255], 2)
                cv2.circle(frame_copy, (top_right[0], top_right[1]), 5, [0, 255, 255], 2)
                cv2.circle(frame_copy, (top_left[0], top_left[1]), 5, [0, 255, 255], 2)
                cv2.circle(frame_copy, (bottom_right[0], bottom_right[1]), 5, [0, 255, 255], 2)
                cv2.circle(frame_copy, (bottom_left[0], bottom_left[1]), 5, [0, 255, 255], 2)
                cv2.imshow('Chest_Camera', frame_copy)  # 显示图像
                #cv2.imshow('chest_red_mask', Imask)
                cv2.waitKey(100)

        else:
            print("815L  chest NONE")



        # 决策执行动作
        angle_ok_flag = False

        if step == 0:   # 前进依据chest 调整大致位置，方向  看底边线调整角度
        
            if top_angle > 2:  # 需要左转
                if top_angle > 6:
                    print("826L 大左转一下  turn001L ",bottom_angle)
                    action_append("turn001L")
                else:
                    print("829L bottom_angle > 3 需要小左转 turn001L ",bottom_angle)
                    action_append("turn001L")
            elif top_angle < -2:  # 需要右转
                if top_angle < -6:
                    print("833L 右大旋转  turn001R < -6 ")
                    action_append("turn001R")
                else:
                    print("836L bottom_angle < -3 需要小右转 turn001R ",bottom_angle)
                    action_append("turn001R")
            elif -2 <= top_angle <= 2:  # 角度正确
                print("839L 角度合适")

                if topcenter_x > 250 or topcenter_x < 230:
                    if topcenter_x > 250:
                        print("843L 微微右移,",topcenter_x)
                        action_append("Right3move")
                    elif topcenter_x < 230:
                        print("846L 微微左移,",topcenter_x)
                        action_append("Left3move")

                else:
                    print("850L 位置合适")
                    break

#找到两个门的轮廓
def find_two(list):
    List_new = []
    a , b = (list[0][0],list[1][0]) if list[0][0] > list[1][0] else (list[1][0],list[0][0])
    for i in range(2,len(list)):
        if list[i][0] > list[0][0]:
            b = a
            a = list[i]
        elif list[i][0] > list[1][0]:
            b =list[i]
    List_new.append(a)
    List_new.append(b)

    return List_new


# ###################### 过   独   木   桥-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
def Greenbridge(colorMask):
    global state_sel, org_img, step, reset, skip, debug, chest_ret

    r_w = chest_r_width
    r_h = chest_r_height

    step = 0
    state = 6

    print("/-/-/-/-/-/-/-/-/-进入Greenbridge")

    while (state == 6):  # 初始化

        # 开始处理图像
        chest_copy = np.rot90(ChestOrg_img)

        chest_copy = chest_copy.copy()
        # chest
        cv2.rectangle(chest_copy, (0, 0), (480, 150), (255, 255, 255), -1)
        border = cv2.copyMakeBorder(chest_copy, 12, 12, 16, 16, borderType=cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255))  # 扩展白边，防止边界无法识别
        Chest_img_copy = cv2.resize(border, (r_w, r_h), interpolation=cv2.INTER_CUBIC)  # 将图片缩放

        Chest_frame_gauss = cv2.GaussianBlur(Chest_img_copy, (3, 3), 0)  # 高斯模糊
        Chest_frame_hsv = cv2.cvtColor(Chest_frame_gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
        Chest_frame_green = cv2.inRange(Chest_frame_hsv, color_range[colorMask][0],
                                        color_range[colorMask][1])  # 对原图像和掩模(颜色的字典)进行位运算
        if img_debug:
            cv2.imshow("mask", Chest_frame_green)
        Chest_opened = cv2.morphologyEx(Chest_frame_green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算 去噪点
        Chest_closed = cv2.morphologyEx(Chest_opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算 封闭连接

        _, Chest_contours, hierarchy = cv2.findContours(Chest_closed, cv2.RETR_LIST,
                                                        cv2.CHAIN_APPROX_NONE)  # 找出轮廓cv2.CHAIN_APPROX_NONE
        # print("Chest_contours len:",len(Chest_contours))
        Chest_areaMaxContour, Chest_area_max = getAreaMaxContour1(Chest_contours)  # 找出最大轮廓
        Chest_percent = round(Chest_area_max * 100 / (r_w * r_h), 2)

        if Chest_areaMaxContour is not None:
            found = 1
            Chest_rect = cv2.minAreaRect(Chest_areaMaxContour)
            # center, w_h, Head_angle = rect  # 中心点 宽高 旋转角度
            Chest_box = np.int0(cv2.boxPoints(Chest_rect))  # 点的坐标

            # 初始化四个顶点坐标
            Chest_top_left = Chest_areaMaxContour[0][0]
            Chest_top_right = Chest_areaMaxContour[0][0]
            Chest_bottom_left = Chest_areaMaxContour[0][0]
            Chest_bottom_right = Chest_areaMaxContour[0][0]
            for c in Chest_areaMaxContour:  # 遍历找到四个顶点
                if c[0][0] + 1.5 * c[0][1] < Chest_top_left[0] + 1.5 * Chest_top_left[1]:
                    Chest_top_left = c[0]
                if (r_w - c[0][0]) + 1.5 * c[0][1] < (r_w - Chest_top_right[0]) + 1.5 * Chest_top_right[1]:
                    Chest_top_right = c[0]
                if c[0][0] + 1.5 * (r_h - c[0][1]) < Chest_bottom_left[0] + 1.5 * (r_h - Chest_bottom_left[1]):
                    Chest_bottom_left = c[0]
                if c[0][0] + 1.5 * c[0][1] > Chest_bottom_right[0] + 1.5 * Chest_bottom_right[1]:
                    Chest_bottom_right = c[0]
            angle_top =  math.atan(
                (Chest_top_right[1] - Chest_top_left[1]) / (Chest_top_right[0] - Chest_top_left[0])) * 180.0 / math.pi
            angle_bottom = math.atan((Chest_bottom_right[1] - Chest_bottom_left[1]) / (
                    Chest_bottom_right[0] - Chest_bottom_left[0])) * 180.0 / math.pi
            Chest_top_center_x = int((Chest_top_right[0] + Chest_top_left[0]) / 2)
            Chest_top_center_y = int((Chest_top_right[1] + Chest_top_left[1]) / 2)
            Chest_bottom_center_x = int((Chest_bottom_right[0] + Chest_bottom_left[0]) / 2)
            Chest_bottom_center_y = int((Chest_bottom_right[1] + Chest_bottom_left[1]) / 2)
            Chest_center_x = int((Chest_top_center_x + Chest_bottom_center_x) / 2)
            Chest_center_y = int((Chest_top_center_y + Chest_bottom_center_y) / 2)
            if img_debug:
                cv2.drawContours(Chest_img_copy, [Chest_box], 0, (0, 0, 255), 2)  # 将大矩形画在图上
                cv2.circle(Chest_img_copy, (Chest_top_right[0], Chest_top_right[1]), 5, [0, 255, 255], 2)
                cv2.circle(Chest_img_copy, (Chest_top_left[0], Chest_top_left[1]), 5, [0, 255, 255], 2)
                cv2.circle(Chest_img_copy, (Chest_bottom_right[0], Chest_bottom_right[1]), 5, [0, 255, 255], 2)
                cv2.circle(Chest_img_copy, (Chest_bottom_left[0], Chest_bottom_left[1]), 5, [0, 255, 255], 2)
                cv2.circle(Chest_img_copy, (Chest_top_center_x, Chest_top_center_y), 5, [0, 255, 255], 2)
                cv2.circle(Chest_img_copy, (Chest_bottom_center_x, Chest_bottom_center_y), 5, [0, 255, 255], 2)
                cv2.circle(Chest_img_copy, (Chest_center_x, Chest_center_y), 7, [255, 255, 255], 2)
                cv2.line(Chest_img_copy, (Chest_top_center_x, Chest_top_center_y),
                         (Chest_bottom_center_x, Chest_bottom_center_y), [0, 255, 255], 2)  # 画出上下中点连线
            if math.fabs(Chest_top_center_x - Chest_bottom_center_x) <= 1:  # 得到连线的角度
                Chest_angle = 90
            else:
                Chest_angle = - math.atan((Chest_top_center_y - Chest_bottom_center_y) / (
                        Chest_top_center_x - Chest_bottom_center_x)) * 180.0 / math.pi
        else:
            Chest_angle = 90
            # center_x = 0.5*r_w
            Chest_center_x = -1
            Chest_bottom_center_x = -1
            Chest_bottom_center_y = -1
            Chest_top_center_x = -1
            Chest_top_center_y = -1

            angle_top = 90
            angle_bottom = 90
            found = 0

            # if step==0:
            # head_angle_dis()

        if img_debug:
            cv2.drawContours(Chest_img_copy, Chest_contours, -1, (255, 0, 255), 1)
            cv2.putText(Chest_img_copy, 'Chest_percent:' + str(Chest_percent) + '%', (30, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
            cv2.putText(Chest_img_copy, "Chest_angle:" + str(int(Chest_angle)), (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
            cv2.putText(Chest_img_copy, "Chest_bottom_center(x,y): " + str(int(Chest_bottom_center_x)) + " , " + str(
                int(Chest_bottom_center_y)), (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
            cv2.putText(Chest_img_copy,
                        "Chest_top_center(x,y): " + str(int(Chest_top_center_x)) + " , " + str(int(Chest_top_center_y)),
                        (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
            cv2.putText(Chest_img_copy, "angle_top:" + str(int(angle_top)), (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 0), 2)  # (0, 0, 255)BGR
            cv2.putText(Chest_img_copy, "angle_bottom:" + str(int(angle_bottom)), (30, 165), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
            cv2.putText(Chest_img_copy, "step :" + str(int(step)), (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),
                        2)  # (0, 0, 255)BGR
            cv2.imshow('Chest_Camera', Chest_img_copy)  # 显示图像
            # cv2.imshow('chest_green_mask', Chest_closed)  # 显示图像
            cv2.waitKey(100)

        # 决策执行动作

        # step=0：
        # 前后：Chest_bottom_center_y
        # 转向：angle_bottom
        # 左右：Chest_bottom_center_x
        if step == 0:  # 接近 看下边沿  角度  Chest_percent > 5
            if found == 0 or Chest_percent < 0.1:
                print("1000L step=0 什么也没有看到，向左转90° turn005L")
                if real_test:
                    action_append("turn001L")
                    time.sleep(sleep_time_s)

            elif Chest_percent > 16 and Chest_bottom_center_y > 460:
                print("1006L step=0, 上桥了")
                step = 1

            elif angle_bottom > 5:
                if angle_bottom > 8:
                    print("1011L step=0 大左转一下 > 8  turn001L angle_bottom={}".format(angle_bottom))
                    print(f"right({Chest_bottom_right[0]},{Chest_bottom_right[1]})  left({Chest_bottom_left[0]},{Chest_bottom_left[1]})")
                    if real_test:
                        action_append("turn001L")
                        time.sleep(sleep_time_s)
                        # if Chest_bottom_center_x > 260 and Chest_bottom_center_y < 400:
                        #     print("1016L 再向右移一些 Right3move angle_bottom={}".format(angle_bottom))
                        #     action_append("Right3move")
                else:
                    print("1019L step=0 小左转 turn000L angle_bottom={}".format(angle_bottom))
                    if real_test:
                        action_append("turn000L")
                # time.sleep(1)
            elif angle_bottom < -5:
                if angle_bottom < -8:
                    print("1666L step=0 大右转一下 < -8  turn001R angle_bottom={}".format(angle_bottom))
                    print(f"right({Chest_bottom_right[0]},{Chest_bottom_right[1]})  left({Chest_bottom_left[0]},{Chest_bottom_left[1]})")
                    if real_test:
                        action_append("turn001R")
                        time.sleep(sleep_time_s)
                        # if Chest_bottom_center_x < 200 and Chest_bottom_center_y < 400:
                        #     print("1030L 再向左移一些 Right3move angle_bottom={}".format(angle_bottom))
                        #     action_append("Left3move")
                else:
                    print("1033L step=0 小右转 turn000R angle_bottom={}".format(angle_bottom))
                    if real_test:
                        action_append("turn000R")
                # time.sleep(1)

            elif Chest_bottom_center_x > 260:  # 右移    center_x
                print("1039L 向右移 Right3move  x>260 Chest_bottom_center_x={}".format(Chest_bottom_center_x))
                if real_test:
                    action_append("Right3move")
            elif Chest_bottom_center_x < 200:  # 左移  center_x
                print("1043L 向左移 Left3move x<200 Chest_bottom_center_x={}".format(Chest_bottom_center_x))
                if real_test:
                    action_append("Left3move")

            elif Chest_bottom_center_y < 460:
                if Chest_bottom_center_y < 330 and 200 <= Chest_bottom_center_x <= 260:
                    print(
                        "1049L y<350 step=0 快速前进 fastForward03 Chest_bottom_center_y={}".format(Chest_bottom_center_y))
                    if real_test:
                        action_append("fastForward03")
                else:
                    print("1053L y<460 step=0 大步前进 两步 Forwalk01 Chest_bottom_center_y={}".format(Chest_bottom_center_y))
                    if real_test:
                        action_append("Forwalk01")
                        time.sleep(sleep_time_s)

                        if angle_bottom >= 3:
                            action_append("turn001L")
                        elif angle_bottom <= -3:
                            action_append("turn001R")

                        action_append("fastForward03")
                        time.sleep(sleep_time_l)
                        if angle_bottom >= 3:
                            action_append("turn001L")
                        elif angle_bottom <= -3:
                            action_append("turn001R")

            elif 220 <= Chest_bottom_center_x <= 240:  # Chest_bottom_center_y < 450
                # print("1071L 前进两步 forwardSlow0403")
                # action_append("forwardSlow0403")
                print("1073L 快走333 fastForward03")
                if real_test:
                    time.sleep(sleep_time_s)
                    action_append("fastForward03")
                    action_append("turn001R")
                    if angle_bottom > 3:
                        action_append("turn001L")
                        action_append("Left1move")
                    elif angle_bottom < -3:
                        action_append("turn001R")
                        action_append("Right1move")

            else:
                print("1086L step = 0 已经到达绿桥边缘，需要进入下一步对准绿桥")
                step = 1
            # 260< Chest_bottom_center_y <460


        elif step == 1:  # 到绿桥边沿，对准绿桥阶段
            if Chest_bottom_center_y > 565:
                print("1096L step = 1, 已经冲到第二阶段了")
                step = 2
            elif angle_bottom > 2:
                if angle_bottom > 6:
                    print("1100L 大左转一下 > 6  turn001L ", angle_bottom)
                    if real_test:
                        action_append("turn001L")
                else:
                    print("1104L 小左转 turn000L ", angle_bottom)
                    if real_test:
                        action_append("turn000L")
                # time.sleep(1)
            elif angle_bottom < -2:
                if angle_bottom < -6:
                    print("1110L 大右转一下 < -6  turn001R ", angle_bottom)
                    if real_test:
                        action_append("turn001R")
                else:
                    print("1114L 小右转 turn001R ", angle_bottom)
                    if real_test:
                        action_append("turn001R")
                # time.sleep(1)
            elif Chest_bottom_center_x > 260:  # 右移    center_x
                print("1119L 向右移 Right02move  x>250")
                if real_test:
                    action_append("Right02move")
            elif Chest_bottom_center_x < 220:  # 左移  center_x
                print("1123L 向左移 Left02move x<220")
                if real_test:
                    action_append("Left02move")
            else:
                print("1127L 对准 快走 Forwalk01")
                if real_test:
                    action_append("Forwalk01")
                    # action_append("turn001R")



        elif step == 2:  # 已经在独木桥阶段  行走独木桥 调整角度 位置  看中线 角度
            if Chest_percent > 2 and Chest_top_center_y > 360:
                print("1136L step = 2, 接近独木桥中点啦，进入第三阶段")
                step = 3
            elif Chest_percent < 2:
                print("1139L step = 2, 接近独木桥终点啦，进入第四阶段")
                action_append("fastForward03")
                action_append("fastForward03")
                step = 4

            elif Chest_bottom_center_x >= 247:  # 右移    center_x
                if Chest_bottom_center_x >= 280:
                    print(
                        "1146L step =2 接近左边缘， 先往右大移  Right3move Chest_bottom_center_x={}".format(Chest_bottom_center_x))
                    if real_test:
                        action_append("Right3move")

                    if Chest_bottom_center_x >= 300:
                        print("1151L step = 2 可能是方向偏左了， 再往右转 turn001R Chest_bottom_center_x={}".format(
                            Chest_bottom_center_x))
                        if real_test:
                            action_append("turn001R")
                elif Chest_bottom_center_x >= 260:
                    print(
                        "1155L step =2 接近左边缘， 先往右移  Right02move Chest_bottom_center_x={}".format(Chest_bottom_center_x))
                    if real_test:
                        action_append("Right02move")
                else:
                    print("1159L step =2 再向右小移 Right1move  Chest_bottom_center_x={}".format(Chest_bottom_center_x))
                    if real_test:
                        action_append("Right1move")

            elif Chest_bottom_center_x <= 213:  # 左移  center_x
                # print("1164L 向左移 Left02move <230 ,", Chest_bottom_center_x)
                if Chest_bottom_center_x <= 180:
                    print(
                        "1166L step =2 接近右边缘， 先往左大移  Left3move Chest_bottom_center_x={}".format(Chest_bottom_center_x))
                    if real_test:
                        action_append("Left3move")

                elif Chest_bottom_center_x <= 200:
                    print(
                        "1171L step =2 接近右边缘， 先往左移  Left02move Chest_bottom_center_x={}".format(Chest_bottom_center_x))
                    if real_test:
                        action_append("Left02move")
                else:
                    print("1175L step =2 再向左小移 Left1move Chest_bottom_center_x={} ".format(Chest_bottom_center_x))
                    if real_test:
                        # action_append("Left02move")
                        action_append("Left1move")

            elif Chest_percent > 2 and Chest_top_center_y > 100:
                # 调整角度位置
                if 0 < Chest_angle < 88:  # 右转
                    if Chest_angle < 87:
                        print("1184L step =2 向右转 turn001R Chest_angle:", Chest_angle)
                        if real_test:
                            action_append("turn000R")
                    else:
                        print("1188L step =2 向右小转 turn000R Chest_angle:", Chest_angle)
                        if real_test:
                            action_append("turn000R")
                        # time.sleep(1)   # timefftest
                elif -88 < Chest_angle < 0:  # 左转
                    if Chest_angle > -87:
                        print("1194L step =2 向左转 turn001L Chest_angle:", Chest_angle)
                        if real_test:
                            action_append("turn000L")
                    else:
                        print("1198L step =2 向左小转 turn000L Chest_angle:", Chest_angle)
                        if real_test:
                            action_append("turn000L")
                        # time.sleep(1)   # timefftest


                else:  # 走三步
                    # print("337L 前进一步 forwardSlow0403")
                    # action_append("forwardSlow0403")
                    print("1207L step =2 上桥后，快走 fastForward03 Ccenter_y:", Chest_center_x)
                    if real_test:
                        action_append("fastForward03")
                        # time.sleep(sleep_time_l)
                        action_append("turn001R")

                        # if abs(Chest_angle - 90) < 2:
                        #     print("暴走")
                        #     action_append("fastForward03")
                        if 0 < Chest_angle < 87:
                            print("1217L 歪了，右转， turn001R Chest_angle={}".format(Chest_angle))
                            if real_test:
                                action_append("turn001R")
                        elif -87 < Chest_angle < 0:
                            print("1221L 歪了，左转， turn001L Chest_angle={}".format(Chest_angle))
                            if real_test:
                                action_append("turn001L")


            else:
                # print("341L 没有看到绿桥向前直行 forwardSlow0403")
                # action_append("forwardSlow0403")
                print("1229L 已经下桥")
                step = 3


        elif step == 3:  # 接近 看上边沿  调整角度  Chest_percent > 5
            if Chest_percent < 1 or Chest_top_center_y > 500:
                # print("1235L 接近桥终点 直行两步离开桥 forwardSlow0403")
                # action_append("forwardSlow0403")
                # action_append("forwardSlow0403")
                # action_append("forwardSlow0403")
                # action_append("Stand")

                print("1241L 接近桥终点 快走离开桥 fastForward03 * 2")
                if real_test:
                    # action_append("fastForward03")
                    action_append("forwardSlow0403")
                    action_append("forwardSlow0403")
                    action_append("forwardSlow0403")
                    action_append("forwardSlow0403")
                    action_append("Stand")

                    step = 4

            elif Chest_top_center_x > 250:  # 右移    center_x
                if Chest_top_center_x > 260:
                    print("1255L step = 3 向右移一大步 Right02move")
                    if real_test:
                        action_append("Right02move")
                else:
                    print("1259L 向右移  >250")
                    if real_test:
                        action_append("Right1move")
            elif Chest_top_center_x < 220:  # 左移  center_x
                if Chest_top_center_x < 200:
                    print("1264L step = 3 向左移一大步 Left02move")
                    if real_test:
                        action_append("Left02move")
                else:
                    print("1268L 向左移 <220")
                    if real_test:
                        action_append("Left1move")

            elif angle_top > 3:
                if angle_top > 7:
                    print("1274L 大左转一下  turn001L angle_top={}".format(angle_top))
                    if real_test:
                        action_append("turn001L")
                else:
                    print("1278L 左转 turn000L angle_top")
                    if real_test:
                        action_append("turn000L")
            elif angle_top < -3:
                if angle_top < -7:
                    print("1283L 大右转一下  turn001R angle_top={}".format(angle_top))
                    if real_test:
                        action_append("turn001R")
                else:
                    print("1287L 右转 turn000R")
                    if real_test:
                        action_append("turn000R")
            elif 220 <= Chest_top_center_x <= 250:
                # print("1802L 前进一步 forwardSlow0403")
                # action_append("forwardSlow0403")
                print("1293L 快走 fastforwardstep")
                if real_test:
                    action_append("fastForward03")


        elif step == 4:  # 离开独木桥阶段   chest 出现bridge  依据chest调整角度位置
            print("1299L 离开桥")

            print("1301L 过桥结束，step = -1  下一关 踢球")
            step = 100

            print("--continue---")
            break

# ###################### 过            门-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
door_flag = True
Angle = 0
angle_top = 0
Bottom_center_y = 0
Bottom_center_x = 0
Top_center_x = 0
Top_center_y = 0
Top_lenth = 0
camera_choice = "Head"


def door_act_move():
    global step, state, reset, skip
    global door_flag
    global real_test
    global camera_choice
    global Angle, angle_top, Bottom_center_y, Bottom_center_x, Top_center_y, Top_center_x, Top_lenth   
    
    step0_far = 130
    step0_close = 24
    step0_angle_top_R = -8
    step0_angle_top_L = 8
    step0_top_center_x_L = 365
    step0_top_center_x_R = 315
    step0_delta = 30 
    step0_turn_times = 3

    step1_angle_top_L = 3
    step1_angle_top_R = -3
    step1_head_bottom_x_F = 265
    step1_head_bottom_x_B = 305
    step1_delta = 30
    step1_close = 375

    step2_get_close = 5

    if step == 0:  # 接近 看下边沿  角度  Chest_percent > 5
        if door_flag == False:
            print("1346L step=0 什么也没有看到，向左转45° turn005L")
            if real_test:
                action_append("turn005L")
                time.sleep(sleep_time_s)

        elif Top_center_y > 160:
            print("1352L step = 0 距离门很远， 快走靠近 fastForward03 Top_center_y={} > 150".format(Top_center_y))
            if real_test:
                action_append("fast_forward_step")
                action_append("turn001R")
                action_append("fast_forward_step")
                time.sleep(sleep_time_l)

        elif Top_center_y > step0_far:
            print("1360L step = 0 再往前一些，慢走 fast_forward_step Top_center_y={} > {}".format(Top_center_y, step0_far))
            if real_test:
                action_append("fast_forward_step")
                time.sleep(sleep_time_l)

        elif Top_center_y < step0_close:
            print("1366L step = 0 距离门很近了, 后退一点 Back3Run Top_center_y={} < {}".format(Top_center_y, step0_close))
            if real_test:
                action_append("Back3Run")
                time.sleep(sleep_time_l)

        elif angle_top < step0_angle_top_R:
            print("1372L step = 0 方向偏了， 向左转 turn001L  angel_top = {} < {}".format(angle_top, step0_angle_top_R))
            if real_test:
                action_append("turn001L")

        elif angle_top > step0_angle_top_L:
            print("1377L step = 0 方向偏了， 向右转 turn001R  angel_top = {} > {}".format(angle_top, step0_angle_top_L))
            if real_test:
                action_append("turn001R")

        elif Top_center_x > step0_top_center_x_L:
            if Top_center_x > step0_top_center_x_L + step0_delta:
                print("1383L step = 0 站位很偏了， 向右移， Right3move Top_center_x = {} > {}".format(Top_center_x, step0_top_center_x_L+step0_delta))
                if real_test:
                    action_append("Right3move")
                    time.sleep(sleep_time_s)
            else:
                print("1388L step = 0 站位偏了， 向右移， Right2move Top_center_x = {} > {}".format(Top_center_x, step0_top_center_x_L))
                if real_test:
                    action_append("Right02move")
                    time.sleep(sleep_time_s)
        elif Top_center_x < step0_top_center_x_R:
            if Top_center_x < step0_top_center_x_R - step0_delta:
                print("1394L step = 0 站位很偏了， 向左移， Left3move Top_center_x = {} < {}".format(Top_center_x, step0_top_center_x_R - step0_delta))
                if real_test:
                    action_append("Left3move")
                    time.sleep(sleep_time_s)
            else:
                print("1399L step = 0 站位偏了， 向左移， Left02move Top_center_x = {} < {}".format(Top_center_x, step0_top_center_x_R))
                if real_test:
                    action_append("Left02move")
                    time.sleep(sleep_time_s)

        else:
            print("1405L 进入下一阶段， 调整侧身 turn005R x {} HeadTurn185".format(step0_turn_times))
            # cv2.waitKey(0)
            if real_test:
                for i in range(0, step0_turn_times):
                    action_append("turn005R")
                    time.sleep(sleep_time_l)

                # action_append("turn004R")
                # action_append("turn001R")
                # action_append("turn001R")
                action_append("HeadTurn185")
                time.sleep(sleep_time_l)
            step = 1

    elif step == 1:
        if Top_lenth < 100:
            print("1421L 歪了！ 左转， 再向右移")
            if real_test:
                action_append("Back3Run")
                action_append("Right02move")

        elif angle_top > step1_angle_top_L or 0 < Angle < 85:
            print("1427L step = 1, 方向偏了， 向右转 turn000R angle_top={} > {}".format(angle_top, step1_angle_top_L))
            if real_test:
                action_append("turn001R")
                time.sleep(sleep_time_l)
        elif angle_top < step1_angle_top_R or -85 < Angle < 0 :
            print("1432L step = 1 方向偏了， 向左转 turn000L angle_top={} < {}".format(angle_top, step1_angle_top_R))
            if real_test:
                action_append("turn001L")
                time.sleep(sleep_time_l)
        
        elif Bottom_center_x < step1_head_bottom_x_F:
            if Bottom_center_x < step1_head_bottom_x_F - step1_delta:
                print("1439L step = 1 站位很靠前了，向后移 Back3Run Bottom_center_x={} < {}".format(Bottom_center_x, step1_head_bottom_x_F - step1_delta))
                if real_test:
                    action_append("Back3Run")
                    time.sleep(sleep_time_s)
            else:
                print("1444L step = 1 站位靠前了，向后移 Back3Run Bottom_center_x={} < {}".format(Bottom_center_x, step1_head_bottom_x_F))
                if real_test:
                    action_append("Back3Run")
                    time.sleep(sleep_time_s)
        
        elif Bottom_center_x > step1_head_bottom_x_B:
            if Bottom_center_x > step1_head_bottom_x_B + step1_delta:
                print("1451L step = 1 站位很靠后了，向前移 Forwalk01 Bottom_center_x={} > {}".format(Bottom_center_x, step1_head_bottom_x_B + step1_delta))
                if real_test:
                    action_append("Forwalk01")
                    time.sleep(sleep_time_s)
            else:
                print("1456L step = 1 站位靠后了，向前移 Forwalk01 Bottom_center_x={} > {}".format(Bottom_center_x, step1_head_bottom_x_B))
                if real_test:
                    action_append("Forwalk01")
                    time.sleep(sleep_time_s)

        elif Bottom_center_y < step1_close:
            print("1462L step = 1, 靠近门, Left3move Bottom_center_y={} < {}".format(Bottom_center_y, step1_close))
            if real_test:
                action_append("Left3move")
                time.sleep(sleep_time_l)
        
        elif Bottom_center_y > step1_close:
            print("1468L 已经接近门了，进入下一阶段，摸黑过门, Bottom_center_y = {} > {}".format(Bottom_center_y, step1_close))
            step = 2

    elif step == 2:
        print("-------/////////////////过门 Left3move x 4")
        # action_append("Back3Run")
        for i in range(0, step2_get_close):
            if real_test:
                action_append("Left3move")
                time.sleep(sleep_time_l)
        # print("向后退一点！ Back3Run")
        # if real_test:
        #     action_append("Back3Run")

        # cv2.waitKey(0)

        for i in range(0, 7):
            if real_test:
                action_append("Left3move")
                if i==3:
                    action_append("turn001R")
                    action_append("turn001R")
                time.sleep(sleep_time_l)

        # cv2.waitKey(0)

        print("完成! ")

        if real_test:
            for i in range(0, step0_turn_times):
                action_append("turn005L")
                time.sleep(sleep_time_l)
            action_append("HeadTurnMM")
            action_append("fast_forward_step")
        
        state = -1


def into_the_door():
    global state_sel, org_img, step, reset, skip, debug, chest_ret, HeadOrg_img, state
    global door_flag
    global camera_choice
    global Angle, angle_top, Bottom_center_y, Bottom_center_x, Top_center_x, Top_center_y, Top_lenth
    step = 0
    state = 5


    r_w = chest_r_width
    r_h = chest_r_height
    

    print("/-/-/-/-/-/-/-/-/-开始过门")

    while(state == 5):
        Area = []
        if camera_choice == "Chest":
            # print("胸部相机")
            chest_OrgFrame = np.rot90(ChestOrg_img)
            Img_copy = chest_OrgFrame.copy()

        elif camera_choice == "Head":
            # print("头部相机")
            Img_copy = HeadOrg_img.copy()
            # Img_copy = cv2.resize(border, (r_w, r_h), interpolation=cv2.INTER_CUBIC)
            # Img_copy = Head_OrgFrame
                    
    
        Frame_gauss = cv2.GaussianBlur(Img_copy, (3, 3), 0)  # 高斯模糊
        Frame_hsv = cv2.cvtColor(Frame_gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
        if camera_choice == "Chest":
            Frame_blue = cv2.inRange(Frame_hsv, color_range['chest_blue_door'][0],
                                        color_range['chest_blue_door'][1])  # 对原图像和掩模(颜色的字典)进行位运算
        elif camera_choice == "Head":
            Frame_blue = cv2.inRange(Frame_hsv, color_range['head_blue_door'][0],
                                        color_range['head_blue_door'][1])  # 对原图像和掩模(颜色的字典)进行位运算
        Opened = cv2.morphologyEx(Frame_blue, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))  # 开运算 去噪点
        Closed = cv2.morphologyEx(Opened, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # 闭运算 封闭连接
        Closed = cv2.dilate(Closed, np.ones((5, 5), np.uint8), iterations=3)
        if img_debug:
            cv2.imshow("Imask", Closed)

        _, contours, hierarchy = cv2.findContours(Closed, cv2.RETR_LIST,
                                                        cv2.CHAIN_APPROX_NONE)  # 找出轮廓cv2.CHAIN_APPROX_NONE

        if len(contours) == 0:
            print("没有找到门！")
            door_flag = False
        
        else:
            door_flag = True
            for i in range(0,len(contours)):
                #print("len[Chest_contours]={}——i:{}".format(len(Chest_contours), i))
                area = cv2.contourArea(contours[i])
                if 2000 < area < 640 * 480 * 0.45:
                    Area.append((area,i))
                
                # print("area{} = {}".format(i, area))
                # cv2.imshow("Processed", Img_copy)
                # cv2.waitKey(0)
            # cv2.drawContours(Img_copy, contours, -1, (0, 0, 255), 1)

            AreaMaxContour, Area_max = getAreaMaxContour1(contours)


            if step != 2 and camera_choice == "Head":
                Rect = cv2.minAreaRect(AreaMaxContour)
                Box = np.int0(cv2.boxPoints(Rect))

                cv2.drawContours(Img_copy, [Box], -1, (255, 200, 100), 2)

                Top_left = AreaMaxContour[0][0]
                Top_right = AreaMaxContour[0][0]
                Bottom_left = AreaMaxContour[0][0]
                Bottom_right = AreaMaxContour[0][0]
                for c in AreaMaxContour:  # 遍历找到四个顶点
                    if c[0][0] + 1.5 * c[0][1] < Top_left[0] + 1.5 * Top_left[1]:
                        Top_left = c[0]
                    if (r_w - c[0][0]) + 1.5 * c[0][1] < (r_w - Top_right[0]) + 1.5 * Top_right[1]:
                        Top_right = c[0]
                    if c[0][0] + 1.5 * (r_h - c[0][1]) < Bottom_left[0] + 1.5 * (r_h - Bottom_left[1]):
                        Bottom_left = c[0]
                    if c[0][0] + 1.5 * c[0][1] > Bottom_right[0] + 1.5 * Bottom_right[1]:
                        Bottom_right = c[0]

                angle_top = - math.atan(
                    (Top_right[1] - Top_left[1]) / (Top_right[0] - Top_left[0])) * 180.0 / math.pi

                Top_lenth = abs(Top_right[0] - Top_left[0])
                Top_center_x = int((Top_right[0] + Top_left[0]) / 2)
                Top_center_y = int((Top_right[1] + Top_left[1]) / 2)
                Bottom_center_x = int((Bottom_right[0] + Bottom_left[0]) / 2)
                Bottom_center_y = int((Bottom_right[1] + Bottom_left[1]) / 2)

                cv2.circle(Img_copy, (Top_right[0], Top_right[1]), 5, [0, 255, 255], 2)
                cv2.circle(Img_copy, (Top_left[0], Top_left[1]), 5, [0, 255, 255], 2)
                cv2.circle(Img_copy, (Bottom_right[0], Bottom_right[1]), 5, [0, 255, 255], 2)
                cv2.circle(Img_copy, (Bottom_left[0], Bottom_left[1]), 5, [0, 255, 255], 2)
                cv2.circle(Img_copy, (Top_center_x, Top_center_y), 5, [0, 255, 255], 2)
                cv2.circle(Img_copy, (Bottom_center_x, Bottom_center_y), 5, [0, 255, 255], 2)
                cv2.line(Img_copy, (Top_center_x, Top_center_y),
                         (Bottom_center_x, Bottom_center_y), [0, 255, 255], 2)  # 画出上下中点连线
                
                if math.fabs(Top_center_x - Bottom_center_x) <= 1:  # 得到连线的角度
                    Angle = 90
                else:
                    Angle = - math.atan((Top_center_y - Bottom_center_y) / (
                            Top_center_x - Bottom_center_x)) * 180.0 / math.pi


                if img_debug:
                    cv2.putText(Img_copy, "angle_top:" + str(int(angle_top)), (30, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 0, 255), 2)
                    cv2.putText(Img_copy, "Head_bottom_center(x,y): " + str(int(Bottom_center_x)) + " , " + str(
                    int(Bottom_center_y)), (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)  # (0, 0, 255)BGR
                    cv2.putText(Img_copy,
                            "Head_top_center(x,y): " + str(int(Top_center_x)) + " , " + str(int(Top_center_y)),
                            (30, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)  # (0, 0, 255)BGR
                    cv2.putText(Img_copy, "Angle:" + str(int(Angle)), (30, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 255), 2)  # (0, 0, 255)BGR
                    cv2.putText(Img_copy, "Top_lenth:" + str(int(Top_lenth)), (400, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)  # (0, 0, 255)BGR

        if img_debug:
            cv2.imshow("Processed", Img_copy)
            cv2.waitKey(10)

        door_act_move()
        print("state={}".format(state))

                

        # if len(Area) > 2:
        #     Area = find_two(Area)
        
        # elif len(Area) < 2:
        #     door_found = False
        #     print("没有发现门框,调用头部相机")
        #     camera_choice = "Head"
        #     # cv2.drawContours(Img_copy, Chest_contours[Area[0][1]], -1, (0, 0, 255), 1)

        # if len(Area) == 2:
        #     door_found = True
        #     Chest_rect1 = cv2.minAreaRect(Chest_contours[Area[0][1]])
        #     Chest_box1 = np.int0(cv2.boxPoints(Chest_rect1))
        #     Chest_rect2 = cv2.minAreaRect(Chest_contours[Area[1][1]])
        #     Chest_box2 = np.int0(cv2.boxPoints(Chest_rect2))

        #     cv2.drawContours(Img_copy, [Chest_box1], -1, (255, 200, 100), 2)
        #     cv2.drawContours(Img_copy, [Chest_box2], -1, (255, 200, 100), 2)

        #     Chest_top_left1 = Chest_contours[Area[0][1]][0][0]
        #     Chest_top_right1 = Chest_contours[Area[0][1]][0][0]
        #     Chest_bottom_left1 = Chest_contours[Area[0][1]][0][0]
        #     Chest_bottom_right1 = Chest_contours[Area[0][1]][0][0]
        #     for c in Chest_contours[Area[0][1]]:  # 遍历找到四个顶点
        #         if c[0][0] + 1.5 * c[0][1] < Chest_top_left1[0] + 1.5 * Chest_top_left1[1]:
        #             Chest_top_left1 = c[0]
        #         if (r_w - c[0][0]) + 1.5 * c[0][1] < (r_w - Chest_top_right1[0]) + 1.5 * Chest_top_right1[1]:
        #             Chest_top_right1 = c[0]
        #         if c[0][0] + 1.5 * (r_h - c[0][1]) < Chest_bottom_left1[0] + 1.5 * (r_h - Chest_bottom_left1[1]):
        #             Chest_bottom_left1 = c[0]
        #         if c[0][0] + 1.5 * c[0][1] > Chest_bottom_right1[0] + 1.5 * Chest_bottom_right1[1]:
        #             Chest_bottom_right1 = c[0]
        #     cv2.circle(Img_copy, (Chest_top_right1[0], Chest_top_right1[1]), 5, [0, 255, 255], 2)
        #     cv2.circle(Img_copy, (Chest_top_left1[0], Chest_top_left1[1]), 5, [0, 255, 255], 2)
        #     cv2.circle(Img_copy, (Chest_bottom_right1[0], Chest_bottom_right1[1]), 5, [0, 255, 255], 2)
        #     cv2.circle(Img_copy, (Chest_bottom_left1[0], Chest_bottom_left1[1]), 5, [0, 255, 255], 2)
        #     angle_Right = - math.atan(
        #     (Chest_top_left1[1] - Chest_bottom_left1[1]) / (Chest_top_left1[0] - Chest_bottom_left1[0])) * 180.0 / math.pi


        #     Chest_top_left2 = Chest_contours[Area[1][1]][0][0]
        #     Chest_top_right2 = Chest_contours[Area[1][1]][0][0]
        #     Chest_bottom_left2 = Chest_contours[Area[1][1]][0][0]
        #     Chest_bottom_right2 = Chest_contours[Area[1][1]][0][0]
        #     for c in Chest_contours[Area[1][1]]:  # 遍历找到四个顶点
        #         if c[0][0] + 1.5 * c[0][1] < Chest_top_left2[0] + 1.5 * Chest_top_left2[1]:
        #             Chest_top_left2 = c[0]
        #         if (r_w - c[0][0]) + 1.5 * c[0][1] < (r_w - Chest_top_right2[0]) + 1.5 * Chest_top_right2[1]:
        #             Chest_top_right2 = c[0]
        #         if c[0][0] + 1.5 * (r_h - c[0][1]) < Chest_bottom_left2[0] + 1.5 * (r_h - Chest_bottom_left2[1]):
        #             Chest_bottom_left2 = c[0]
        #         if c[0][0] + 1.5 * c[0][1] > Chest_bottom_right2[0] + 1.5 * Chest_bottom_right2[1]:
        #             Chest_bottom_right2 = c[0]
        #     cv2.circle(Img_copy, (Chest_top_right2[0], Chest_top_right2[1]), 5, [0, 255, 255], 2)
        #     cv2.circle(Img_copy, (Chest_top_left2[0], Chest_top_left2[1]), 5, [0, 255, 255], 2)
        #     cv2.circle(Img_copy, (Chest_bottom_right2[0], Chest_bottom_right2[1]), 5, [0, 255, 255], 2)
        #     cv2.circle(Img_copy, (Chest_bottom_left2[0], Chest_bottom_left2[1]), 5, [0, 255, 255], 2)
        #     angle_Left = - math.atan(
        #     (Chest_top_right2[1] - Chest_bottom_right2[1]) / (Chest_top_right2[0] - Chest_bottom_right2[0])) * 180.0 / math.pi
            
        #     Chest_top_center_x = int((Chest_top_right2[0] + Chest_top_left1[0]) / 2)
        #     Chest_top_center_y = int((Chest_top_right2[1] + Chest_top_left1[1]) / 2)
        #     cv2.circle(Img_copy, (Chest_top_center_x, Chest_top_center_y), 5, [0, 255, 255], 2)

        #     cv2.putText(Img_copy, "Chest_top_center(x,y): " + str(int(Chest_top_center_x)) + " , " + str(
        #     int(Chest_top_center_y)), (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        #     cv2.putText(Img_copy, "angle_left:" + str(int(angle_Left)), (30, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        #             (0, 0, 255), 2)  # (0, 0, 255)BGR
        #     cv2.putText(Img_copy, "angle_right:" + str(int(angle_Right)), (30, 460), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.65, (0, 0, 255), 2)



# ###################### 踢            球-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
golf_angle_ball = 90
Chest_ball_angle = 90
hole_Angle = 45
golf_angle = 0
ball_x = 0
ball_y = 0
golf_angle_flag = False
golf_dis_start = True
golf_angle_start = False
golf_ok = False
hole_flag = False
Chest_ball_flag = False
Chest_golf_angle = 0

ball_dis_start = True
hole_angle_start = False

head_state = 0  # 90 ~ -90      左+90   右-90

hole_x = 0
hole_y = 0
jump_count = 0
count = 0
angle_dis_count = 0
picnum = 0
fast_run = True

###################################################踢球决策
def kick_act_move():
    global step, state, reset, skip
    global hole_Angle, ball_hole
    global golf_angle_ball, golf_angle, Chest_ball_angle, Chest_golf_angle
    global ball_x, ball_y, Chest_ball_x, Chest_ball_y
    global golf_angle_flag, golf_dis_flag  # golf_dis_flag未使用
    global golf_angle_start
    global golf_ok
    global hole_flag, Chest_ball_flag
    global ball_dis_start, hole_angle_start
    global head_state, angle_dis_count, fast_run
    global count
    global jump_count
    ball_hole_angle_ok = False

    # 由脚底到红球延伸出一条射线，依据球洞与该射线的关系，调整机器人位置
    # ball_hole_local()

    if True:
        if step == -1:
            # for i in range(0, 2):
            #     if edge_angle('red_floor') == 1:
            #         action_append("Forwalk01")

            # action_append("HeadTurnMM")
            # action_append("Right3move")
            # action_append("Right3move")
            action_append("Stand")
            action_append("Forwalk01")

            action_append("Stand")
            action_append("turn001L")
            # action_append("Forwalk01")
            # action_append("turn001L")
            action_append("Forwalk01")
            action_append("Forwalk01")
            action_append("turn001R")
            step = 0
        # step = 0 # 单步调试某一步骤用
        elif step == 0:  # 发现球，发现球洞，记录球与球洞的相对位置
            # print("看黑线调整居中")
            if Chest_ball_flag is True:  # 前进到球跟前
                if fast_run:
                    if Chest_ball_y <= 320:  # 340
                        print("2002L step = 0 看到了球，距离很远, 快走前进 fastForward04 Chest_ball_y={} < 320".format(Chest_ball_y))
                        # action_append("forwardSlow0403")
                        # action_append("forwardSlow0403")
                        if real_test:
                            action_append("Forwalk01")
                            action_append("turn001R")
                            time.sleep(sleep_time_l)
                            # head_angle_dis()  # headfftest
                    elif Chest_ball_y <= 400:  # 340
                        print("2012L 看到了球，距离远， 快走前进 Forwalk01 Chest_ball_y={} < 290".format(Chest_ball_y))
                        if real_test:
                            action_append("Forwalk01")
                            time.sleep(sleep_time_l)
                            # head_angle_dis()    # headfftest

                    if hole_flag:
                        if 45 < hole_Angle < 70:
                            print(
                                "2020L step = 0,看到球门了，向球靠近的过程中，也要对准球门，以免看不到球门了 Hole_angle={}, 因此需要向右转 turn001R".format(
                                    hole_Angle))
                            if real_test:
                                action_append("turn001R")
                        elif -70 < hole_Angle < -45:
                            print(
                                "2024L step = 0,看到球门了，向球靠近的过程中，也要对准球门，以免看不到球门了 Hole_angle={}, 因此需要向左转 turn001L".format(
                                    hole_Angle))
                            if real_test:
                                action_append("turn001L")

                    if Chest_ball_y > 370:
                        print("2029L 已接近球了,不能再跑这么快了，进入细调模式（ball_y > 270） Chest_ball_y={}".format(Chest_ball_y))
                        fast_run = False

                else:
                    if Chest_ball_y < 370:  # 390 400改成了390 zzx 10.14
                        # X
                        if Chest_ball_x < 140*(4/3):  # 240 - 100
                            print("2036L step = 0 Chest_ball_x < 180 左侧移 Chest_ball_x={}".format(Chest_ball_x))
                            if real_test:
                                action_append("Left3move")
                        elif Chest_ball_x > 340*(4/3):  # 240 + 100
                            print("2040L step = 0 Chest_ball_x > 300 右侧移 Chest_ball_x={}".format(Chest_ball_x))
                            if real_test:
                                action_append("Right3move")
                        else:
                            if Chest_ball_y < 370:
                                print("2045L step = 0 再靠球近些（ball_y < 370）前挪一点 forwalkVeryslow Chest_ball_y={}".format(
                                    Chest_ball_y))
                                if real_test:
                                    # action_append("Forwalk01") #zzx 10.14
                                    action_append("forwalkVeryslow")
                                    action_append("turn001R")
                                # action_append("Forwalk02")
                            else:
                                print("2052L step = 0 再靠球近些（ball_y < 360）前挪一点点  Chest_ball_y={}".format(Chest_ball_y))
                                if real_test:
                                    action_append("forwalkVeryslow")
                                    action_append("Stand")

                    elif Chest_ball_y > 430:  # 470改成了430 zzx 10.14
                        print("2058L step = 0 隔球太近（ball_y < 360）后退一点点  Chest_ball_y={}".format(Chest_ball_y))
                        if real_test:
                            action_append("Stand")
                            action_append("Back3Run")

                    elif Chest_ball_y >= 400 and Chest_ball_y <= 450:  # Chest_ball_y>360
                        print("2063L goto step1  Chest_ball_y={}".format(Chest_ball_y))
                        step = 1
            else:
                print("未发现球 寻找球")
                count += 1
                if real_test:
                    if count > 5:
                        action_append("turn001R")  # ffetst
                        action_append("Stand")
                    # elif count < 1:
                    #     action_append("Forwalk01")
                    else:
                        # action_append("Forwalk01")        zzx 10.13
                        action_append("forwalkVeryslow")
                        action_append("turn001R")

                # if head_state == 0:
                #     print("头右转(-60)寻找球")
                #     head_state = -60
                # elif head_state == -60:
                #     print("头由右转变为左转(+60)寻找球")
                #     head_state = 60
                # elif head_state == 60:
                #     print("头部 恢复0 向前迈进")

        elif step == 1:  # 看球调整位置   逐步前进调整至看球洞
            if Chest_ball_flag is False:
                if real_test:
                    action_append("Stand")
                    action_append("Back3Run")

            elif Chest_ball_y <= 400:
                print("2094L step = 1 前挪一点点 forwalkVeryslow < 380 Chest_ball_y={}".format(Chest_ball_y))
                if real_test:
                    action_append("forwalkVeryslow")
            elif Chest_ball_y > 450:
                print("2098L step = 1 后一步 Back3Run > 480 Chest_ball_y={}".format(Chest_ball_y))
                if real_test:
                    action_append("Stand")
                    action_append("Back3Run")
            elif 400 < Chest_ball_y <= 450:
                if hole_flag == True:
                    if head_state == -60:
                        print("头右看，看到球洞")
                        step = 2
                        # print("172L 头恢复0 向右平移")
                        # head_state = 0
                    elif head_state == 60:
                        print("头左看，看到球洞")
                        step = 3
                        # print("172L 头恢复0 向左平移")
                        # head_state = 0
                    elif head_state == 0:  # 头前看 看到球洞
                        print("2115L step4")
                        step = 4
                else:
                    print("2118L error 左右旋转头 寻找球洞 ")
                    if real_test:
                        action_append("turn001R")

        elif step == 4:  # 粗略调整朝向   球与球洞大致在一条线
            # print("调整红球在左脚正前方不远处，看球洞的位置调整")
            if ball_dis_start:
                if Chest_ball_x <= int(200*(4/3)):
                    if Chest_ball_x < int(200*(4/3)):
                        print("2199L4 step = 4 需要左侧移 Left3move Chest_ball_x={}".format(Chest_ball_x))
                        if real_test:
                            action_append("Left3move")
                    else:
                        print("2203L4 step = 4 需要左侧移 Left02move Chest_ball_x={}".format(Chest_ball_x))
                        if real_test:
                            action_append("Left02move")
                    angle_dis_count = 0
                elif Chest_ball_x > 280*(4/3):
                    if Chest_ball_x > 280*(4/3):
                        print("2209L4 step = 4 需要右侧移 Right3move Chest_ball_x={}".format(Chest_ball_x))
                        if real_test:
                            action_append("Right3move")
                    else:
                        print("2213L4 step = 4 需要右侧移 Right02move Chest_ball_x={}".format(Chest_ball_x))
                        if real_test:
                            action_append("Right02move")
                    angle_dis_count = 0
                else:
                    print("2218L4 Chest_ball_y---位置ok")
                    ball_dis_start = False
                    hole_angle_start = True
            if hole_angle_start:
                if hole_Angle <= 0:
                    # angle
                    if hole_Angle > -70:
                        if hole_Angle >= -65:
                            if Chest_ball_y > 500:
                                print("2227L4 需要后挪一点 Back3Run Chest_ball_y={}".format(Chest_ball_y))
                                if real_test:
                                    action_append("Stand")
                                    action_append("Back3Run")
                                angle_dis_count = 0
                            elif Chest_ball_y < 420:
                                print("2232L4 需要前挪一点 forwalkVeryslow Chest_ball_y={}".format(Chest_ball_y))
                                if real_test:
                                    action_append("forwalkVeryslow")
                                angle_dis_count = 0

                            print("2237L4 大左转一下  turn003L hole_Angle={}".format(hole_Angle))
                            if real_test:
                                action_append("turn003L")
                        else:
                            if Chest_ball_y > 470:
                                print("2242L4 需要后挪一点 Back3Run Chest_ball_y".format(Chest_ball_y))
                                if real_test:
                                    action_append("Stand")
                                    action_append("Back3Run")
                                angle_dis_count = 0
                            elif Chest_ball_y < 440:
                                print("2247L4 需要前挪一点 forwalkVeryslow Chest_ball_y={}".format(Chest_ball_y))
                                if real_test:
                                    action_append("forwalkVeryslow")
                                angle_dis_count = 0

                            print("2252L4 左转一下  turn001L hole_Angle={}".format(hole_Angle))
                            if real_test:
                                action_append("turn001L")
                    else:
                        print("2256L4 hole_Angle---角度ok")
                        angle_dis_count = angle_dis_count + 1
                        ball_dis_start = True
                        hole_angle_start = False

                    # ball_dis_start = True
                    # hole_angle_start = False
                if hole_Angle > 0:
                    # angle
                    if hole_Angle < 70:
                        if hole_Angle <= 65:
                            if Chest_ball_y > 500:
                                print("2268L4 需要后挪一点 Back3Run Chest_ball_y={}".format(Chest_ball_y))
                                if real_test:
                                    action_append("Stand")
                                    action_append("Back3Run")
                                angle_dis_count = 0
                            elif Chest_ball_y < 440:
                                print("2273L4 需要前挪一点 forwalkVeryslow Chest_ball_y={}".format(Chest_ball_y))
                                if real_test:
                                    action_append("forwalkVeryslow")
                                angle_dis_count = 0

                            print("2278L4 大右转一下 turn001R hole_Angle={}".format(hole_Angle))
                            action_append("turn001R")  # turn003R 改成了 turn001R zzx 10.14
                        else:
                            if Chest_ball_y > 500:
                                print("2282L4 需要后挪一点 Back3Run Chest_ball_y={}".format(Chest_ball_y))
                                action_append("Stand")
                                action_append("Back3Run")
                                angle_dis_count = 0
                            elif Chest_ball_y < 440:
                                print("2286L4 需要前挪一点 forwalkVeryslow Chest_ball_y={}".format(Chest_ball_y))
                                if real_test:
                                    action_append("forwalkVeryslow")
                                angle_dis_count = 0

                            print("2291L4 右转一下 turn001R ", hole_Angle)
                            action_append("turn001R")
                    else:
                        print("2294L4 hole_Angle---角度OK")
                        angle_dis_count = angle_dis_count + 1
                        ball_dis_start = True
                        hole_angle_start = False

                    # ball_dis_start = True
                    # hole_angle_start = False

                if angle_dis_count > 1:
                    angle_dis_count = 0
                    print("2304L  step step 5555")
                    step = 5


        elif step == 5:  # 调整 球与球洞在一条直线    球范围  230<Chest_ball_y<250
            # print("55555 球与球洞都在")
            # print("2310L 调整红球在左脚正前方不远处，看球洞的位置调整")
            if ball_dis_start:  # 390<y<450  230<x<250
                if Chest_ball_x == 0:
                    jump_count += 1
                elif Chest_ball_x < 220*(4/3):
                    if Chest_ball_x < 210*(4/3):
                        print("2318L 需要左侧移 Left02move Chest_ball_x={}".format(Chest_ball_x))
                        if real_test:
                            action_append("Left02move")
                    else:
                        print("2322L 需要左侧移一点 Left1move Chest_ball_x={}".format(Chest_ball_x))
                        if real_test:
                            action_append("Left1move")
                    jump_count = 0
                    angle_dis_count = 0
                elif Chest_ball_x > 260*(4/3):
                    # if Chest_ball_x - 240 > 40:
                    #     print("2328L 需要右侧移 Right02move")
                    #     action_append("Right02move")
                    # else:
                    if Chest_ball_x > 270*(4/3):
                        print("2332L 需要右侧移 Right02move Chest_ball_x={}".format(Chest_ball_x))
                        if real_test:
                            action_append("Right02move")
                    else:
                        print("2336L 需要右侧移一点 Right1move Chest_ball_x={}".format(Chest_ball_x))
                        if real_test:
                            action_append("Right1move")
                    angle_dis_count = 0
                    jump_count = 0
                else:
                    print("2341L Chest_ball_y---位置ok")
                    ball_dis_start = False
                    hole_angle_start = True
                    jump_count = 0
            if hole_angle_start:
                if hole_Angle < 0:
                    # angle
                    if hole_Angle > -70:
                        # y
                        if Chest_ball_y > 515:
                            print("2350L 需要后挪一点 Back3Run Chest_ball_y={}".format(Chest_ball_y))
                            if real_test:
                                action_append("Stand")
                                action_append("Back3Run")
                            angle_dis_count = 0
                        elif Chest_ball_y < 460:
                            print("2355L 需要前挪一点 forwalkVeryslow  Chest_ball_y={}".format(Chest_ball_y))
                            if real_test:
                                action_append("forwalkVeryslow")
                            angle_dis_count = 0

                        if hole_Angle >= -65:
                            print("2361L 大左转一下  turn001L hole_Angle={}".format(hole_Angle))
                            if real_test:
                                action_append("turn001L")
                        else:
                            print("2365L 左转一下  turn001L ", hole_Angle)
                            if real_test:
                                action_append("turn001L")
                    else:
                        print("2369L hole_Angle---角度ok")
                        angle_dis_count = angle_dis_count + 1

                    ball_dis_start = True
                    hole_angle_start = False
                if hole_Angle > 0:
                    # angle
                    if hole_Angle < 70:
                        # y
                        if Chest_ball_y > 515:
                            print("2379L 需要后挪一点 Back3Run Chest_ball_y={}".format(Chest_ball_y))
                            if real_test:
                                action_append("Stand")
                                action_append("Back3Run")
                            angle_dis_count = 0
                        elif Chest_ball_y < 480:
                            print("2384L 需要前挪一点 forwalkVeryslow Chest_ball_y={}".format(Chest_ball_y))
                            if real_test:
                                action_append("forwalkVeryslow")
                            angle_dis_count = 0

                        if hole_Angle <= 65:
                            print("2390L 大右转一下 turn001R hole_Angle={}".format(hole_Angle))
                            if real_test:
                                action_append("turn001R")
                        else:
                            print("2394L 右转一下 turn001R hole_Angle={}".format(hole_Angle))
                            if real_test:
                                action_append("turn001R")
                    else:
                        print("2398L hole_Angle---角度OK")
                        angle_dis_count = angle_dis_count + 1

                    ball_dis_start = True
                    hole_angle_start = False

                if jump_count > 20:
                    jump_count = 0
                    step = 9

                if angle_dis_count > 1:
                    angle_dis_count = 0
                    step = 6


        elif step == 6:
            # print("666")
            if Chest_ball_angle > 75 and hole_Angle > 75:
                ball_hole_angle_ok = True
            if Chest_ball_angle < -75 and hole_Angle > 75:
                ball_hole_angle_ok = True
            if Chest_ball_angle < -75 and hole_Angle < -75:
                ball_hole_angle_ok = True
            if Chest_ball_angle > 75 and hole_Angle < -75:
                ball_hole_angle_ok = True

            if Chest_ball_angle > 73 and hole_Angle > 73 and ball_hole_angle_ok == False:
                print("2421L 右转一点点 turn001R")
                if real_test:
                    action_append("turn001R")
            elif Chest_ball_angle < -73 and hole_Angle < -73 and ball_hole_angle_ok == False:
                print("2425L 左转一点点 turn001L")
                if real_test:
                    action_append("turn001L")
            elif Chest_ball_y <= 460:
                print("2429L 向前挪动一点点 forwalkVeryslow")
                if real_test:
                    action_append("forwalkVeryslow")
                    # action_append("turn001R")

            elif hole_x > 250*(4/3):
                print("step = 6  方向偏左了, 往右转 turn001R hole_x={}".format(hole_x))
                if real_test:
                    action_append("turn001R")
            else:
                print("/////////////////////////////next step 进入最后对准阶段 step=7")
                step = 7

        elif step == 7:
            if Chest_ball_y > 515:
                print("2444L 靠太近了，向后挪动一点点 Back0Run Chest_ball_y={} > 500".format(Chest_ball_y))
                if real_test:
                    action_append("Stand")
                    action_append("Back3Run")

            # elif 80 < Chest_ball_angle < 85:
            #     print("2449L 右转一点点 turn000R")
            #     if real_test:
            #         action_append("turn000R")

            elif Chest_ball_x > 203*(4/3):  # 210
                if Chest_ball_x > 220*(4/3):
                    print("2455L step = 7 向右移动 Right02move Chest_ball_x={} > 200".format(Chest_ball_x))
                    if real_test:
                        action_append("Right02move")
                        time.sleep(sleep_time_s)
                else:
                    print("2460L step = 7 向右移动一点点 Right1move Chest_ball_x = {} > 195".format(Chest_ball_x))
                    if real_test:
                        action_append("Right1move")
                        time.sleep(sleep_time_s)
            elif Chest_ball_x < 180*(4/3):
                if Chest_ball_x < 175*(4/3):
                    print("2466L step = 7 向左移动 Left02move Chest_ball_x={} < 175".format(Chest_ball_x))
                    if real_test:
                        action_append("Left02move")
                        time.sleep(sleep_time_s)
                else:
                    print("2471L 向左移动 Left1move")
                    if real_test:
                        action_append("Left1move")
                        time.sleep(sleep_time_s)
            elif Chest_ball_y < 470:
                print("2476L  step = 7 向前挪动一点点 forwalkVeryslow Chest_ball_y={} < 490".format(Chest_ball_y))
                if real_test:
                    action_append("forwalkVeryslow")
                    time.sleep(sleep_time_l)
                    action_append("turn000R")

            # elif hole_Angle >80:
            #     print("站位有问题，后退， 重整 Back3Run")
            #     step = 5
            #     if real_test:
            #         action_append("Back3Run")

            else:
                print("2490L 踢球踢球阶段 LfootShot")
                step = 8


        elif step == 8:
            if real_test:
                if Chest_ball_y > 515:
                    print("向后退")
                    action_append("Stand")
                    action_append("Back3Run")

                elif Chest_ball_angle > 80 or Chest_ball_angle <= 0:
                    if Chest_ball_angle <= 0 or Chest_ball_angle > 82:
                        print("右移一大步")
                        action_append("Right02move")
                    else:
                        print("右移一小步")
                        action_append("Right1move")

                elif Chest_ball_angle < 77:
                    if Chest_ball_angle < 75:
                        print("左移一大步")
                        action_append("Left02move")
                    else:
                        print("左移一小步")
                        action_append("Left1move")

                elif hole_Angle < -80 or hole_Angle > 0:
                    print("右转左移")
                    action_append("turn001R")
                    action_append("Stand")
                    action_append("Left1move")

                elif hole_Angle > -75:
                    print("左转右移")
                    action_append("turn001L")
                    action_append("Stand")
                    action_append("Right1move")

                elif Chest_ball_y < 490:
                    print("2527L  step = 7 向前挪动一点点 Forwalk00 Chest_ball_y={} < 490".format(Chest_ball_y))
                    if real_test:
                        action_append("forwalkVeryslow")
                        time.sleep(sleep_time_l)
                        # action_append("turn000R")

                else:
                    print("准备踢球")
                    action_append("Stand")
                    time.sleep(0.5)
                    action_append("LfootShot")
                    step = 9
                    if real_test:
                        action_append("turn005L")
                        action_append("turn005L")
                        action_append("turn005L")
                        action_append("turn005L")
                        action_append("turn005L")
                        # action_append("Forwalk01")


        elif step == 9:
            for i in range(0, 3):
                if edge_angle('red_floor') == 1:
                    action_append("Forwalk01")
                    if i == 1:
                        action_append("Right3move")

            action_append("HeadTurnMM")
            action_append("Right3move")
            action_append("Right3move")
            action_append("Forwalk01")
            action_append("fastorward_step")
            print("完成！ 77777")
            state = -1
            step = 10
 

def kick_ball():
    global state, state_sel, step, reset, skip
    global hole_Angle
    global golf_angle_ball, golf_angle, Chest_ball_angle, Chest_golf_angle
    global ball_x, ball_y, Chest_ball_x, Chest_ball_y
    global hole_flag, Chest_ball_flag
    global ChestOrg_img
    global picnum, img_debug

    # 初始化
    sum_contours = np.array([[[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]])
    step = -1
    state = 7

    while state == 7:
        if -1 <= step < 9:  # 踢球的七步
            ChestOrg = ChestOrg_img.copy()
            ChestOrg = np.rot90(ChestOrg)
            HeadOrg = HeadOrg_img.copy()
            Hole_OrgFrame = HeadOrg.copy()
            Hole_OrgFrame = cv2.resize(Hole_OrgFrame, (int(640), int(640)))
            Ball_OrgFrame = ChestOrg.copy()
            Ball_OrgFrame = cv2.resize(Ball_OrgFrame, (int(640), int(640)))

            img_h, img_w = Hole_OrgFrame.shape[:2]

            # 把上中心点和下中心点200改为640/2  fftest
            bottom_center = (int(320), int(img_h))  # 图像底中点
            top_center = (int(320), int(0))  # 图像顶中点
            # bottom_center = (int(640/2), int(img_h))  #图像底中点
            # top_center = (int(640/2), int(0))     #图像顶中点

            # 开始处理图像
            Hole_hsv = cv2.cvtColor(Hole_OrgFrame, cv2.COLOR_BGR2HSV)

            Hole_Imask = cv2.inRange(Hole_hsv, color_range['blue_hole'][0], color_range['blue_hole'][1])    # 识别到洞
            Hole_Imask = cv2.dilate(Hole_Imask, np.ones((5, 5), np.uint8), iterations=3)
            Hole_Imask = cv2.erode(Hole_Imask, np.ones((3, 3), np.uint8), iterations=3)

            # 初始化
            hole_center = [0, 0]
            Chest_ball_center = [0, 0]

            temp = 100
            temp_e = None
            temp_i = -1
            temp_area = 0

            temp_b = 100
            temp_b_e = None
            temp_b_i = -1
            temp_b_area = 0

            # chest 球洞处理
            hole_x = 0
            hole_y = 0

            _, cnts, hierachy = cv2.findContours(Hole_Imask, cv2.RETR_CCOMP,
                                                 cv2.CHAIN_APPROX_NONE)  # **获得图片轮廓值  #遍历图像层级关系
            # *取得一个球洞的轮廓*
            for i in range(0, len(cnts)):  # 初始化sum_contours，使其等于其中一个c，便于之后拼接的格式统一
                # cv2.drawContours(Hole_OrgFrame, cnts[i], -1, (0, 0, 255), 1)
                # cv2.imshow("Contours", Hole_OrgFrame)
                # cv2.waitKey(0)
                area = cv2.contourArea(cnts[i])  # 计算轮廓面积
                # print(len(cnts))
                # print("area={}".format(area))
                # if img_debug and area > 100:
                #     cv2.putText(Hole_OrgFrame, "area:" + str(area), (10, Hole_OrgFrame.shape[0] - 55),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
                if 640 * 480 * 0.0033 < area < 640 * 480 * 0.45:  # 去掉很小的干扰轮廓以及最大的图像边界
                    e = cv2.fitEllipse(cnts[i])     # 拟合椭圆，获得ellipse =  [ (x, y) , (a, b), angle ]。（x, y）代表椭圆中心点的位置；
                                                    # （a, b）代表长短轴长度，应注意a、b为长短轴的直径，而非半径；angle 代表了中心旋转的角度
                    area2 = np.pi * e[1][0] * e[1][1]
                    # print("ratio:{}".format(area/area2))
                    if area / area2 > 0.05 and np.abs(90 - e[2]) < 90:      # 不太懂这个判断条件的意义
                        if temp < e[0][1]:
                            temp = e[0][1]
                            temp_e = e
                            temp_i = i
                            temp_area = area
                        else:
                            continue
                    # break
                else:
                    continue

            if temp_i == -1:
                print("没有找到洞")
                hole_flag = False
            else:
                cnt_large = cnts[temp_i]
                cv2.ellipse(Hole_OrgFrame, temp_e, (255, 255, 255), 1)
                hole_flag = True
                (hole_x, hole_y), radius = cv2.minEnclosingCircle(cnt_large)  # 最小内接圆形
                hole_center = (int(hole_x), int(hole_y))
                radius = int(radius)
                cv2.circle(Hole_OrgFrame, hole_center, radius, (100, 200, 30), 2)
                # ellipse = cv2.fitEllipse(cnt_large)
                # cv2.ellipse(OrgFrame,ellipse,(255,255,0),2)
                cv2.line(Hole_OrgFrame, hole_center, bottom_center, (0, 0, 100), 2)
                if (hole_center[0] - bottom_center[0]) == 0:
                    hole_Angle = 90
                else:
                    # hole_Angle  (y1-y0)/(x1-x0)
                    hole_Angle = - math.atan(
                        (hole_center[1] - bottom_center[1]) / (hole_center[0] - bottom_center[0])) * 180.0 / math.pi

            if img_debug:
                cv2.putText(Hole_OrgFrame, "step:" + str(step),
                            (10, Hole_OrgFrame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(Hole_OrgFrame, "hole_angle:" + str(hole_Angle),
                            (10, Hole_OrgFrame.shape[0] - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(Hole_OrgFrame, "hole_x:" + str(hole_x),
                            (10, Hole_OrgFrame.shape[0] - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(Hole_OrgFrame, "hole_y:" + str(hole_y),
                            (220, Hole_OrgFrame.shape[0] - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(Hole_OrgFrame, "hole_flag:" + str(hole_flag),
                            (10, Hole_OrgFrame.shape[0] - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            # chest 红球处理
            Chest_ball_x = 0
            Chest_ball_y = 0
            # 模板匹配，远距离靠近
            if step == -2:      # 该部分没有办法进入
                template = cv2.imread('//home//pi//RunningRobot_test//template.jpg')
                w = template.shape[0]
                h = template.shape[1]

                meth = 'cv2.TM_SQDIFF_NORMED'
                method = eval(meth)
                res = cv2.matchTemplate(Ball_OrgFrame, template, method)
                min_val, max_val, top_left, max_loc = cv2.minMaxLoc(res)
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(Ball_OrgFrame, top_left, bottom_right, 255, 2)
                Chest_ball_x = int(top_left[0] + w / 2)
                Chest_ball_y = int(top_left[1] + h / 2)
                Chest_ball_flag = True

            else:
                if step < 4:
                    e_kernelSize = 3
                else:
                    e_kernelSize = 5


                Chest_Ball_hsv = cv2.cvtColor(Ball_OrgFrame, cv2.COLOR_BGR2HSV)
                # Chest_Ball_hsv = cv2.GaussianBlur(Chest_Ball_hsv, (3, 3), 0)

                # Chest_Ball_Imask_1 = cv2.inRange(Chest_Ball_hsv, color_range['d_red_ball_floor1'][0],
                #                                  color_range['d_red_ball_floor1'][1])
                # Chest_Ball_Imask_2 = cv2.inRange(Chest_Ball_hsv, color_range['d_red_ball_floor2'][0],
                #                                  color_range['d_red_ball_floor2'][1])
                # Chest_Ball_Imask = cv2.bitwise_or(Chest_Ball_Imask_1, Chest_Ball_Imask_2)
                Chest_Ball_Imask = cv2.inRange(Chest_Ball_hsv,color_range['kick_ball_rec'][0],color_range['kick_ball_rec'][1])
                Chest_Ball_Imask = cv2.erode(Chest_Ball_Imask, np.ones((e_kernelSize, e_kernelSize), np.uint8), iterations=2)
                Chest_Ball_Imask = cv2.morphologyEx(Chest_Ball_Imask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8),iterations=1)

                # cv2.imshow("red_floor_INV", Chest_Ball_Imask)
                # cv2.waitKey(0)
                # Chest_Ball_Imask = cv2.inRange(Chest_Ball_hsv, color_range['ball_red'][0], color_range['ball_red'][1])

                # Chest_Ball_Imask = cv2.erode(Chest_Ball_Imask, None, iterations=5)
                # Chest_Ball_Imask = cv2.dilate(Chest_Ball_Imask, np.ones((7, 7), np.uint8), iterations=2)
                # cv2.imshow("red_ball", Chest_Ball_Imask)
                # cv2.waitKey(0)

                _, cnts2, hierachy2 = cv2.findContours(Chest_Ball_Imask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts2 is not None:
                    for i in range(0, len(cnts2)):
                        area = cv2.contourArea(cnts2[i])  # 计算轮廓面积
                        if img_debug:
                            # print(len(cnts2))
                            # print("area={}".format(area))
                            if area > 100:
                                cv2.putText(Ball_OrgFrame, "area:" + str(area), (10, Ball_OrgFrame.shape[0] - 55),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
                        if 300 < area < 640 * 480 * 0.025:  # 去掉很小的干扰轮廓以及最大的图像边界
                            if cnts2[i].size > 10:
                                e = cv2.fitEllipse(cnts2[i])
                                area2 = np.pi * e[1][0] * e[1][1]
                                # print("e={} 等待".format(e))
                                # print("ratio:{}".format(area/area2))
                                bias = abs(1 - e[1][0] / e[1][1])
                                # cv2.waitKey(0)
                                if ((step < 4 and area / area2 > 0.05 and 290 < e[0][1] < 550) or (step >= 4 and 400 < e[0][1] < 550)) and \
                                        (e[1][1]/e[1][0] < 2.3):
                                    if temp_b > bias:
                                        temp_b = bias
                                        temp_b_e = e
                                        temp_b_i = i
                                        temp_b_area = area

                                    else:
                                        continue
                            else:
                                continue
                            # cv2.waitKey(0)

                            # break
                        else:
                            # cv2.drawContours(Hole_OrgFrame, cnts, -1, (0, 0, 255), 3)
                            continue
                else:
                    print("2887L cnt_large is None")
                    continue

                # 圆球轮廓  计算角度 Chest_ball_angle
                if temp_b_i == -1:
                    print("没有找到球")
                    # cv2.waitKey(0)
                    Chest_ball_flag = False
                    Chest_ball_y = 0
                    Chest_ball_x = 0
                else:
                    print("球位置：{} 面积：{}".format(temp_b_e, temp_b_area))
                    cnt_large3 = cnts2[temp_b_i]
                    if img_debug:
                        cv2.ellipse(Ball_OrgFrame, temp_b_e, (255, 255, 255), 1)
                    Chest_ball_flag = True
                    (Chest_circle_x, Chest_circle_y), Chest_radius = cv2.minEnclosingCircle(cnt_large3)
                    Chest_ball_center = (int(Chest_circle_x), int(Chest_circle_y))
                    Chest_radius = int(Chest_radius)
                    if img_debug:
                        cv2.circle(Ball_OrgFrame, Chest_ball_center, Chest_radius, (100, 200, 20), 2)
                        cv2.line(Ball_OrgFrame, Chest_ball_center, top_center, (0, 100, 0), 2)
                    # ellipse = cv2.fitEllipse(cnt_large)
                    # cv2.ellipse(OrgFrame,ellipse,(255,255,0),2)
                    if (Chest_ball_center[0] - top_center[0]) == 0:
                        Chest_ball_angle = 90
                    else:
                        # *Chest_ball_angle*  (y1-y0)/(x1-x0)
                        Chest_ball_angle = - math.atan((Chest_ball_center[1] - top_center[1]) / (
                                Chest_ball_center[0] - top_center[0])) * 180.0 / math.pi

                    Chest_ball_x = int(Chest_circle_x)  # *ball_x*
                    Chest_ball_y = int(Chest_circle_y)  # *ball_y*

            if img_debug:
                cv2.putText(Ball_OrgFrame, "step:" + str(step),
                            (10, Ball_OrgFrame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(Ball_OrgFrame, "Chest_ball_x:" + str(Chest_ball_x),
                            (10, Ball_OrgFrame.shape[0] - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(Ball_OrgFrame, "Chest_ball_y:" + str(Chest_ball_y),
                            (220, Ball_OrgFrame.shape[0] - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(Ball_OrgFrame, "Chest_ball_flag:" + str(Chest_ball_flag),
                            (10, Hole_OrgFrame.shape[0] - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(Ball_OrgFrame, "ball_angle:" + str(Chest_ball_angle),
                            (10, Ball_OrgFrame.shape[0] - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        else:
            break

        if img_debug:
            cv2.imshow("Ball_OrgFrame", Ball_OrgFrame)
            cv2.imshow("Hole_OrgFrame", Hole_OrgFrame)
            cv2.waitKey(10)
        kick_act_move()


###################### 档            板-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
def baffle():
    global org_img, step, reset, skip
    global handling
    print("/-/-/-/-/-/-/-/-/-进入baffle")
    step = 0
    baffle_dis_Y_flag = False
    baffle_angle = 0
    notok = True
    see = False
    finish = False
    angle = 45
    dis = 0
    dis_flag = False
    angle_flag = False
    center_x = 0
    while(1):
        if True:
            Corg_img = ChestOrg_img.copy()
            Corg_img = np.rot90(Corg_img)
            # Corg_img = Corg_img[int(300):int(400),int(100):int(500)]
            OrgFrame = Corg_img.copy()
            handling = Corg_img.copy()
            frame = Corg_img.copy()
            center = []

    # 开始处理图像
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
            Imask = cv2.inRange(hsv, color_range['blue_baf'][0], color_range['blue_baf'][1])
            Imask = cv2.erode(Imask, None, iterations=2)
            Imask = cv2.dilate(Imask, np.ones((3, 3), np.uint8), iterations=2)
            # cv2.imshow('BLcolor', Imask)
            _, cnts, hieracy = cv2.findContours(Imask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓
            # print("cnts len:",len(cnts))
            if cnts is not None:
                cnt_large , cnt_area = getAreaMaxContour1(cnts)

                #print(cnt_area)
        
            else:
                print("2984L cnt_large is None")
                continue

            blue_bottom_Y = 0
            if cnt_large is not None:
                rect = cv2.minAreaRect(cnt_large)  # 最小外接矩形
                box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点
                
                Ax = box[0, 0]
                Ay = box[0, 1]
                Bx = box[1, 0]
                By = box[1, 1]
                Cx = box[2, 0]
                Cy = box[2, 1]
                Dx = box[3, 0]
                Dy = box[3, 1]
                pt1_x, pt1_y = box[0, 0], box[0, 1]
                pt3_x, pt3_y = box[2, 0], box[2, 1]
                center_x = int((pt1_x + pt3_x) / 2)
                center_y = int((pt1_y + pt3_y) / 2)
                center.append([center_x, center_y])
                cv2.drawContours(OrgFrame, [box], -1, [0, 0, 255, 255], 3)
                cv2.circle(OrgFrame, (center_x, center_y), 10, (0, 0, 255), -1)  # 画出中心点
                # 求得大矩形的旋转角度，if条件是为了判断长的一条边的旋转角度，因为box存储的点的顺序不确定\
                if math.sqrt(math.pow(box[3, 1] - box[0, 1], 2) + math.pow(box[3, 0] - box[0, 0], 2)) > math.sqrt(math.pow(box[3, 1] - box[2, 1], 2) + math.pow(box[3, 0] - box[2, 0], 2)):
                    baffle_angle = - math.atan((box[3, 1] - box[0, 1]) / (box[3, 0] - box[0, 0])) * 180.0 / math.pi
                else:
                    baffle_angle = - math.atan( (box[3, 1] - box[2, 1]) / (box[3, 0] - box[2, 0]) ) * 180.0 / math.pi  # 负号是因为坐标原点的问题
                if center_y > blue_bottom_Y:
                    blue_bottom_Y = center_y
            baffle_dis_Y = blue_bottom_Y
            baffle_dis_X = center_x 
            if baffle_dis_Y > 240:
                baffle_dis_Y_flag = True


            if img_debug:
                cv2.putText(OrgFrame, "baffle_dis_Y:" + str(baffle_dis_Y),
                            (10, OrgFrame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            
                cv2.putText(OrgFrame, "baffle_dis_Y_flag:" + str(baffle_dis_Y_flag),
                            (10, OrgFrame.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            
                cv2.putText(OrgFrame, "baffle_angle:" + str(baffle_angle),
                            (10, OrgFrame.shape[0] - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(OrgFrame, "step:" + str(step), (30, OrgFrame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),2)  # (0, 0, 255)BGR
                            
                cv2.imshow('OrgFrame', OrgFrame)
                k = cv2.waitKey(10)
                if k == 27:
                    cv2.destroyWindow('closed_pic')
                    cv2.destroyWindow('org_img_copy')
                    break
                elif k == ord('s'):
                    print("save picture123")
                    cv2.imwrite("picture123.jpg",org_img) #保存图片

            
    # 决策执行动作
            if step == 0:
                if baffle_dis_Y <= 250:
                    print("3045L 大步前进 Forwalk02")
                    action_append("Forwalk02")
                elif baffle_dis_Y > 250:
                    step=1


            elif step==1:   # 调整角度 -5 ~ 5
                if baffle_angle > 5:
                    if baffle_angle > 8:
                        print("3054L 大左转一下  turn001L  baffle_angle:",baffle_angle)
                        action_append("turn001L")
                    else:
                        print("3057L 左转 turn000L  baffle_angle:",baffle_angle)
                        action_append("turn000L")
                elif baffle_angle < -5:
                    if baffle_angle < -8:
                        print("3061L 大右转一下  turn001R  baffle_angle:",baffle_angle)
                        action_append("turn001R")
                    else:
                        print("3064L 右转 turn000R  baffle_angle:",baffle_angle)
                        action_append("turn000R")
                else:
                    step=2
                
            elif step == 2:     # 调整前进位置  调整左右位置
                if baffle_dis_Y < 390:
                    print("3071L 大一步前进 forwardSlow0403")
                    action_append("forwardSlow0403")
                elif 390 < baffle_dis_Y < 460:
                    print("3074L 向前挪动 Forwalk00")
                    action_append("Forwalk00")
                elif 460 < baffle_dis_Y:
                    step = 3
            elif step == 3: # 调整角度
                if baffle_angle > 2:
                    if baffle_angle > 5:
                        print("3081L 大左转一下  turn001L ",baffle_angle)
                        action_append("turn001L")
                    else:
                        print("3084L 左转 turn001L")
                        action_append("turn001L")
                elif baffle_angle < -2:
                    if baffle_angle < -5:
                        print("3088L 大右转一下  turn001R ",baffle_angle)
                        action_append("turn001R")
                    else:
                        print("3091L 右转 turn001R ",baffle_angle)
                        action_append("turn001R")
                elif baffle_dis_Y_flag:
                    step = 4
            elif step == 4: # 跨栏后调整方向

                print("3097L 前挪一点点")
                print("3098L 翻栏杆 翻栏杆 RollRail")
                action_append("Right3move")
                action_append("Right3move")
                action_append("Right3move")
                action_append("Stand")
                action_append("RollRail")
                action_append("Stand")
                #print("step step step 444 ")
                
                
                action_append("turn004L")
                action_append("turn004L")
                action_append("turn004L")
                action_append("turn001L")
                action_append("Back3Run")
                # action_append("turn004L")
                # action_append("turn004L")
                
                break


###################### 过            坑-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
def hole_edge(color):
    edge_angle_chest(color)#调整好角度与距离
    while(1):
        Area = 0
        src = ChestOrg_img.copy()
        src = np.rot90(src)
        src = src.copy()
        # cv2.imshow("src1",src)
        src = src[int(0):int(400),int(50):int(500)]
        src_copy = src
        # cv2.imshow("src2",src)
        src = cv2.GaussianBlur(src, (5, 5), 0)
        hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv",hsv_img)
        mask = cv2.inRange(hsv_img, color_range[color][0], color_range[color][1])
        # cv2.imshow("mask",mask)


        mask2 = cv2.erode(mask, None, iterations=5)
        mask1 = cv2.dilate(mask2, None, iterations=8)

        # cv2.imshow("mask1",mask1)
        # # cv2.imshow("mask2",mask2)
        # # # cv2.imshow("mask",mask)
        # cv2.waitKey()

        _, contours2, hierarchy2 = cv2.findContours(mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if len(contours2) >= 2:
            print("3146L 仍然看得到内轮廓，向前走 forwardSlow0403")
            action_append("forwardSlow0403")

        else:
            print("已近迈进，正式进入过坑阶段")
            action_append("Stand")
            if  color == 'blue_hole_chest':
                hole_edge_main('blue_hole_head')
                break
            elif color == 'green_hole_chest':
                hole_edge_main('green_hole_head')
                break
        
def hole_edge_main(color):
    global HeadOrg_img,chest_copy, reset, skip,handling
    global handling
    angle_ok_flag = False
    angle = 90
    dis = 0
    bottom_centreX = 0
    bottom_centreY = 0
    see = False
    dis_ok_count = 0
    headTURN = 0
    hole_flag = 0

    step = 1
    print("/-/-/-/-/-/-/-/-/-hole edge")
    while True:
        OrgFrame = HeadOrg_img.copy()
        x_start = 180
        blobs = OrgFrame[int(0):int(480), int(x_start):int(380)]  # 只对中间部分识别处理  Y , X
        handling = blobs.copy()
        frame_mask = blobs.copy()

        # 获取图像中心点坐标x, y
        center = []
        # 开始处理图像
     
        hsv = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        Imask = cv2.inRange(hsv, color_range[color][0], color_range[color][1])
        # Imask = cv2.erode(Imask, np.ones((3, 3), np.uint8), iterations=1)
        Imask = cv2.dilate(Imask, np.ones((3, 3), np.uint8), iterations=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        Imask = cv2.morphologyEx(Imask, cv2.MORPH_OPEN, kernel)
        _, contours, hierarchy = cv2.findContours(Imask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓
        # cv2.imshow("opened",Imask)
        # print("len:",len(cnts))

        if len(contours) > 0: 
            max_area = max(contours, key=cv2.contourArea)
            epsilon = 0.05 * cv2.arcLength(max_area, True)
            approx = cv2.approxPolyDP(max_area, epsilon, True)
            approx_list = list(approx)
            approx_after = []
            for i in range(len(approx_list)):
                approx_after.append(approx_list[i][0])
            approx_sort = sorted(approx_after, key=lambda x: x[1], reverse=True)
            # if approx_sort[0][0] > approx_sort[1][0]:
            #     approx_sort[0], approx_sort[1] = approx_sort[1], approx_sort[0]
            if len(approx_sort) == 4:
                bottom_line = (approx_sort[3], approx_sort[2])
                center_x = (bottom_line[1][0]+bottom_line[0][0])/2
                center_y = (bottom_line[1][1]+bottom_line[0][1])/2
            else:
                bottom_line = None

        else:
            bottom_line = None
            
        # 初始化
        L_R_angle = 0 
        blackLine_L = [0,0]
        blackLine_R = [0,0]

        if bottom_line is not None:
            see = True
            if bottom_line[0][1] - bottom_line[1][1]==0:
                angle=90
            else:
                angle = - math.atan((bottom_line[1][1] - bottom_line[0][1]) / (bottom_line[1][0] - bottom_line[0][0]))*180.0/math.pi
            Ycenter = int((bottom_line[1][1] + bottom_line[0][1]) / 2)
            Xcenter = int((bottom_line[1][0] + bottom_line[0][0]) / 2)
            if bottom_line[1][1] > bottom_line[0][1]:
                blackLine_L = [bottom_line[1][0] , bottom_line[1][1]]
                blackLine_R = [bottom_line[0][0] , bottom_line[0][1]]
            else:
                blackLine_L =  [bottom_line[0][0] , bottom_line[0][1]]
                blackLine_R = [bottom_line[1][0] , bottom_line[1][1]]
            cv2.circle(OrgFrame, (Xcenter + x_start, Ycenter), 10, (255,255,0), -1)#画出中心点

            if blackLine_L[0] == blackLine_R[0]:
                L_R_angle = 0
            else:
                L_R_angle =  (-math.atan( (blackLine_L[1]-blackLine_R[1]) / (blackLine_L[0]-blackLine_R[0]) ) *180.0/math.pi)+4



            if img_debug:
                
                cv2.circle(OrgFrame, (blackLine_L[0] + x_start, blackLine_L[1]), 5, [0, 255, 255], 2)
                cv2.circle(OrgFrame, (blackLine_R[0] + x_start, blackLine_R[1]), 5, [255, 0, 255], 2)
                cv2.line(OrgFrame, (blackLine_R[0] + x_start,blackLine_R[1]), (blackLine_L[0] + x_start,blackLine_L[1]), (0, 255, 255), thickness=2)
                cv2.putText(OrgFrame, "L_R_angle:" + str(L_R_angle),(10, OrgFrame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(OrgFrame, "Xcenter:" + str(Xcenter + x_start),(10, OrgFrame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(OrgFrame, "Ycenter:" + str(Ycenter),(200, OrgFrame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                # cv2.drawContours(frame_mask, cnt_sum, -1, (255, 0, 255), 3)
                cv2.imshow('frame_mask', frame_mask)
                # cv2.imshow('black', Imask)
                cv2.imshow('OrgFrame', OrgFrame)
                cv2.waitKey(10)
        else:
            see = False
            
        #print(Ycenter)

     # 决策执行动作
        if step == 1:
            print("3266L 向右看 HeadTurn015")
            action_append("HeadTurn015")
            time.sleep(1)   # timefftest
            step = 2
        elif step == 2:
            if not see:  # not see the edge
                # cv2.destroyAllWindows()
                print("3273L 右侧看不到边缘 左侧移 Left3move")
                action_append("Left3move")
            else:   # 0
                if L_R_angle > 1.5:
                    if L_R_angle > 7:
                        headTURN += 1
                        print("3279L 左da旋转 turn001L ",L_R_angle)
                        action_append("turn001L")

                    else:
                        print("3283L 左旋转 turn000L ",L_R_angle)
                        headTURN += 1
                        action_append("turn000L")

                    
                    # time.sleep(1)   # timefftest
                elif L_R_angle < -1.5:
                    if L_R_angle < -7:
                        headTURN += 1
                        print("3292L 右da旋转  turn001R ",L_R_angle)
                        action_append("turn001R")

                    else:
                        print("3296L 右旋转  turn000R ",L_R_angle)
                        action_append("turn000R")

                    
                    # time.sleep(1)   # timefftest
                elif Ycenter >= 365:
                    if Ycenter > 390:
                        print("3303L 左da侧移 Left1move >440 ",Ycenter)
                        action_append("Left3move")
                    else:
                        print("3306L 左侧移 Left02move > 365 ",Ycenter)
                        action_append("Left1move")
                elif Ycenter < 355:
                    print("3309L 右侧移 Right02move <400 ",Ycenter)
                    action_append("Right02move")
                else:
                    print("3312L 右看 X位置ok")
                    action_append("Forwalk01")
                    action_append("Forwalk01")
                    action_append("Forwalk01")
                    action_append("Forwalk01")
                    action_append("Forwalk01")
                    action_append("Forwalk01")
                    action_append("Forwalk01")
                    action_append("Forwalk01")
                    action_append("Forwalk01")
                    action_append("Forwalk01")
                    # action_append("Left02move")
                    #print("向前一步")
                    # action_append("forwardSlow0403")
                    # action_append("forwardSlow0403")  
                    # action_append("forwardSlow0403")
                    action_append("Stand")
                    step = 3
                    #cv2.destroyAllWindows()
                 

        elif step == 3:
            edge_angle_chest(color)
            action_append("HeadTurnMM")
            step = 5

        elif step == 5:
            print("过坑阶段结束")
            action_append("Stand")
            break



###################### 过   地   雷   区-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
def angle_adjust():#调整角度，确保始终朝前
    global baffle_angle,Bbox_centerY
    if Bbox_centerY > 250:
        if baffle_angle > 2:
            if baffle_angle > 5:
                print("朝前 大左转一下  turn001L ",baffle_angle)
                action_append("turn001L")
            else:
                print("朝前 左转 turn001L")
                action_append("turn001L")
        elif baffle_angle < -2:
            if baffle_angle < -5:
                print("朝前 大右转一下  turn001R ",baffle_angle)
                action_append("turn001R")
            else:
                print("朝前 右转 turn001R ",baffle_angle)
                action_append("turn001R")
    else:
         pass
        
Bbox_centerY = 0

def area_bits(Imask):
    area=0
    for i in Imask:
        for j in i:
            if j!=0:
                area=area+1
            else:
                continue
    return area
def obstacle():
    global HeadOrg_img, step
    global Head_L_R_angle,Bbox_centerY,blue_rail
    global baffle_angle
    print("/-/-/-/-/-/-/-/-/-进入obscle")
    action_append("Stand")
    step = 1
    k = 1
    blue_rail = False
    fall_right=False
    fall_left =False
    
    while(1):
        if True:    
            if ChestOrg_img is None:
                continue
            Corg_img = ChestOrg_img.copy()
            Corg_img = np.rot90(Corg_img)
            #Corg_img = Corg_img[int(200):int(400),int(100):int(500)]
            Corg_img = Corg_img.copy()
            hsv = cv2.cvtColor(Corg_img, cv2.COLOR_BGR2HSV)
            hsv = cv2.GaussianBlur(hsv, (3, 3), 0)


         # blue 分析图像 决策执行
            Bumask = cv2.inRange(hsv,color_range['blue_baf'][0],color_range['blue_baf'][1])
            Bumask = cv2.erode(Bumask, None, iterations=2)
            Bumask = cv2.dilate(Bumask, np.ones((3, 3), np.uint8), iterations=2)
            # cv2.imshow('Bluemask', Bumask)
            _, cntsblue, hierarchy = cv2.findContours(Bumask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # 找出轮廓
            #添加是否到板子边缘的决策，纠正绕行动作
            Imask_whiteboard=cv2.inRange(hsv,color_range['gray_dir'][0],color_range['gray_dir'][1])
            detect_right=[540,600,320,480]#从左到右是Y1,Y2,X1,X2
            detect_left =[540,600,320,480]
            single_right=Imask_whiteboard[int(detect_right)[0],int(detect_right)[1],int(detect_right)[2],int(detect_right)[3]]
            single_left=Imask_whiteboard[int(detect_left)[0], int(detect_left)[1], int(detect_left)[2], int(detect_left)[3]]
            area_right_detect=area_bits(single_right)
            area_left_detect =area_bits(single_left)
            if area_right_detect<0.5*(detect_right[1]-detect_right[0])*(detect_right[3]-detect_right[2]):
                print("检测到右边是空的，是不是要摔了？")
                fall_right = True
            if area_left_detect <0.5*(detect_left[1]-detect_left[0])*(detect_left[3]-detect_left[2]):
                print("检测到左边是空的，是不是要摔了？")
                fall_left  = True
            
            if cntsblue is not None:
                cnt_large = getAreaMaxContour2(cntsblue)    # 取最大轮廓
            else:
                print("1135L cnt_large is None")
                continue

            if cnt_large is not None:
                rect_blue = cv2.minAreaRect(cnt_large)
                box_blue = np.int0(cv2.boxPoints(rect_blue))  # 点的坐标
                Bbox_centerX = int((box_blue[3,0] + box_blue[2,0] + box_blue[1,0] + box_blue[0,0])/4)
                Bbox_centerY = int((box_blue[3,1] + box_blue[2,1] + box_blue[1,1] + box_blue[0,1])/4)
                Bbox_center = [Bbox_centerX,Bbox_centerY]
                cv2.circle(Corg_img, (Bbox_center[0],Bbox_center[1]), 7, (0, 0, 255), -1) # 圆点标记

                cv2.drawContours(Corg_img, [box_blue], -1, (255,0,0), 3)
                if math.sqrt(math.pow(box_blue[3, 1] - box_blue[0, 1], 2) + math.pow(box_blue[3, 0] - box_blue[0, 0], 2)) > math.sqrt(math.pow(box_blue[3, 1] - box_blue[2, 1], 2) + math.pow(box_blue[3, 0] - box_blue[2, 0], 2)):
                    baffle_angle = - math.atan((box_blue[3, 1] - box_blue[0, 1]) / (box_blue[3, 0] - box_blue[0, 0])) * 180.0 / math.pi
                else:
                    baffle_angle = - math.atan( (box_blue[3, 1] - box_blue[2, 1]) / (box_blue[3, 0] - box_blue[2, 0]) ) * 180.0 / math.pi  # 负号是因为坐标原点的问题
                obscle_area_blue = 0
                # 当遇到蓝色门槛时停止
                for c in cntsblue:
                    obscle_area_blue += math.fabs(cv2.contourArea(c))
                if  Bbox_centerY >= 280 and obscle_area_blue > 0.05 * 640 * 480 :   # and go_up: # 320  obscle_area_blue > 0.05 * 640 * 480 and

                    if img_debug:
                        cv2.imshow('Corg_img', Corg_img)
                        cv2.waitKey(10)
                    print("遇到蓝色门槛-----*-----*-----*-----* Bbox_center Y:",Bbox_centerY)
                    action_append("Stand")
                    blue_rail = True
                    

                    cv2.destroyAllWindows()
                    break

         # black 分析图像 决策执行
            Imask = cv2.inRange(hsv, color_range['black_dir'][0], color_range['black_dir'][1])#黑色地雷
            Imask = cv2.erode(Imask, None, iterations=3)
            Imask = cv2.dilate(Imask, np.ones((3, 3), np.uint8), iterations=2)
            #cv2.imshow('black', Imask)
            _, contours, hierarchy = cv2.findContours(Imask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓
            cv2.drawContours(Corg_img, contours, -1, (255, 0, 255), 2)


            left_point = [640,0]
            right_point = [0,0]

            
            if len(contours) != 0:

                Big_battle = [0,0]

                for c in contours:
                    rect = cv2.minAreaRect(c)  # 最小外接矩形
                    box = cv2.boxPoints(rect)   #我们需要矩形的4个顶点坐标box, 通过函数 cv2.cv.BoxPoints() 获得
                    box = np.intp(box)  # 最小外接矩形的四个顶点
                    box_Ax,box_Ay = box[0,0],box[0,1]
                    box_Bx,box_By = box[1,0],box[1,1]
                    box_Cx,box_Cy = box[2,0],box[2,1]
                    box_Dx,box_Dy = box[3,0],box[3,1]
                    box_centerX = int((box_Ax + box_Bx + box_Cx + box_Dx)/4)
                    box_centerY = int((box_Ay + box_By + box_Cy + box_Dy)/4)
                    box_center = [box_centerX,box_centerY]

                    # 剔除图像上部分点 和底部点
                    if box_centerY < 300 or box_centerY > 550:
                        continue
                    
                    # 遍历点 画圈
                    if box_debug:
                        cv2.circle(Corg_img, (box_centerX,box_centerY), 8, (0, 0, 255), 2) # 圆点标记识别黑点
                        cv2.imshow('Corg_img', Corg_img)
                        cv2.waitKey(1)
                        
                    # 找出最左点与最右点
                    if  box_centerX < left_point[0]:
                        left_point = box_center
                    if box_centerX > right_point[0]:
                        right_point = box_center

                    if box_centerX <= 80 or box_centerX >= 400 :  # 排除左右边沿点 box_centerXbox_centerX 240
                        continue
                    if math.pow(box_centerX - 240 , 2) + math.pow(box_centerY - 640 , 2) < math.pow(Big_battle[0] - 240 , 2) + math.pow(Big_battle[1] - 640 , 2):
                        Big_battle =  box_center  # 这个是要规避的黑点
                        # print("1272L go_up False ",Big_battle[0],Big_battle[1])

                # 显示图
                if img_debug:
                    cv2.circle(Corg_img, (left_point[0],left_point[1]), 7, (0, 255, 0), -1) # 圆点标记
                    cv2.circle(Corg_img, (right_point[0],right_point[1]), 7, (0, 255, 255), -1) # 圆点标记
                    cv2.circle(Corg_img, (Big_battle[0],Big_battle[1]), 7, (255, 255, 0), -1) # 圆点标记
                    cv2.putText(Corg_img, "Bbox_centerY:" + str(int(Bbox_centerY)), (230, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0, 0, 0), 2)  # (0, 0, 255)BGR
                    cv2.putText(Corg_img, "Big_battle x,y:" + str(int(Big_battle[0])) +', ' + str(int(Big_battle[1])) , (230, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0, 0, 0), 2)  # (0, 0, 255)BGR
                    cv2.putText(Corg_img, "baffle_angle:" + str(int(baffle_angle)), (230, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0, 0, 0), 2)  # (0, 0, 255)BGR
                    cv2.line(Corg_img, (Big_battle[0],Big_battle[1]), (240,640), (0, 255, 255), thickness=2)
                    cv2.line(Corg_img, (0,500), (480,500), (255, 255, 255), thickness=2)
                    cv2.rectangle(Corg_img, (50,350), (430,550), (0, 0, 255), thickness=2)
                    
                    # cv2.imshow('handling', handling)
                    cv2.imshow('Corg_img', Corg_img)
                    k = cv2.waitKey(100)
                    if k == 27:
                        cv2.destroyWindow('closed_pic')
                        cv2.destroyWindow('org_img_copy')
                        break
                    elif k == ord('s'):
                        print("save picture123")
                        cv2.imwrite("picture123.jpg",HeadOrg_img) #保存图片

                
                #370修改为360
                if Big_battle[1] < 350:
                    print("3564L 前进靠近一步 forwardSlow0403 ",Big_battle[1])
                    action_append("Stand")
                    action_append("forwardSlow0403")
                    action_append("Stand")
                    angle_adjust()
                   

                #410
                elif Big_battle[1] < 400:
                    print("3575L 慢慢前进靠近 Forwalk01",Big_battle[1])
                    action_append("Stand")
                    action_append("Forwalk01")
                    action_append("Stand")
                    angle_adjust()

              
             
                
                elif (50 <= Big_battle[0] and Big_battle[0] < 140):
                    print("3580L 右平移一步 Right02move", Big_battle[0])
                    if fall_right==True:
                        print("但是右侧好像要摔了，还是往左走吧")
                        action_append("Left3move")
                        action_append('Left3move')
                        continue
                    action_append("Stand")
                    action_append("Right02move")

                    # 240修改为265
                elif (140 <= Big_battle[0] and Big_battle[0] < 240):
                    print("3586L 右平移三步 Right3move", Big_battle[0])
                    if fall_right==True:
                        print("但是右侧好像要摔了，还是往左走吧")
                        action_append("Stand")
                        action_append("Right3move")
                        action_append("Stand")
                        action_append("Right3move")
                        action_append("Stand")
                        action_append("Right02move")
                        continue
                    action_append("Stand")
                    action_append("Right3move")
                    action_append("Stand")
                    action_append("Right02move")
                    action_append("Stand")
                    action_append("Right02move")



                elif (240 <= Big_battle[0] and Big_battle[0] < 360):
                    print("3592L 向左平移三步 Left3move", Big_battle[0])
                    if fall_left==True:
                        print("但是左侧好像要摔了，还是往右走吧")
                        action_append("Stand")
                        action_append("Left3move")
                        action_append("Stand")
                        action_append("Left3move")
                        action_append("Stand")
                        action_append("Left3move")
                        action_append("Stand")
                        action_append("Left02move")
                        continue
                    action_append("Stand")
                    action_append("Left3move")
                    action_append("Stand")
                    action_append("Left3move")
                    action_append("Stand")
                    action_append("Left3move")



                elif (360 <= Big_battle[0] < 430):
                    print("3598L 向左平移一步 Left02move", Big_battle[0])
                    if fall_left==True:
                        print("但是左侧好像要摔了，还是往右走吧")
                        action_append("Left3move")
                        action_append('Left3move')
                        continue
                    action_append("Stand")
                    action_append("Left02move")
                    
                    

                else:
                    print("3604L error 不在范围 继续向前走")
                    action_append("Stand")
                    action_append("forwardSlow0403")
                    # Big_battle = [0,0]
            else:
                print("3607L 继续向前")
                # print(Big_battle)
                action_append("forwardSlow0403")
                Big_battle = [0,0]

                if img_debug:
                    cv2.circle(Corg_img, (left_point[0],left_point[1]), 7, (0, 255, 0), -1) # 圆点标记
                    cv2.circle(Corg_img, (right_point[0],right_point[1]), 7, (0, 255, 255), -1) # 圆点标记
                    cv2.circle(Corg_img, (Big_battle[0],Big_battle[1]), 7, (255, 255, 0), -1) # 圆点标记
                    cv2.line(Corg_img, (Big_battle[0],Big_battle[1]), (240,640), (0, 255, 255), thickness=2)
                    # 500线
                    cv2.line(Corg_img, (0,500), (480,500), (255, 255, 255), thickness=2)
                    cv2.imshow('Corg_img', Corg_img)
                    k = cv2.waitKey(100)
                    if k == 27:
                        cv2.destroyWindow('closed_pic')
                        cv2.destroyWindow('org_img_copy')
                        break
                    elif k == ord('s'):
                        print("save picture123")
                        cv2.imwrite("picture123.jpg",HeadOrg_img) #保存图片        

###################### 终            点-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
def end_door():
    global ChestOrg_img, state, state_sel, step, reset, skip, img_debug,end_door_flag
    end_door_flag = 0
    state_sel = 'door'
    state = 1
    if state == 1:  # 初始化
        print("/-/-/-/-/-/-/-/-/-进入end_door")
        step = 0
    else:
        return

    while state == 1 :

        if step == 0: #判断门是否抬起
            if ChestOrg_img is None:
                continue
            
            org_img_copy = ChestOrg_img.copy()
            org_img_copy = np.rot90(org_img_copy)
            handling = org_img_copy.copy()           

            border = cv2.copyMakeBorder(handling, 12, 12, 16, 16, borderType=cv2.BORDER_CONSTANT,value=(255, 255, 255))     # 扩展白边，防止边界无法识别
            handling = cv2.resize(border, (chest_r_width, chest_r_height), interpolation=cv2.INTER_CUBIC)                   # 将图片缩放
            frame_gauss = cv2.GaussianBlur(handling, (21, 21), 0)       # 高斯模糊
            frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)    # 将图片转换到HSV空间
            
            frame_door_yellow = cv2.inRange(frame_hsv, color_range['yellow_door'][0], color_range['yellow_door'][1])    # 对原图像和掩模(颜色的字典)进行位运算
            frame_door_black = cv2.inRange(frame_hsv, color_range['black_door'][0], color_range['black_door'][1])       # 对原图像和掩模(颜色的字典)进行位运算


            frame_door = cv2.add(frame_door_yellow, frame_door_black)    
            open_pic = cv2.morphologyEx(frame_door, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))      # 开运算 去噪点
            closed_pic = cv2.morphologyEx(open_pic, cv2.MORPH_CLOSE, np.ones((50, 50), np.uint8))   # 闭运算 封闭连接        

            (image, contours, hierarchy) = cv2.findContours(closed_pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓
            areaMaxContour, area_max = getAreaMaxContour1(contours)  # 找出最大轮廓
            percent = round(100 * area_max / (chest_r_width * chest_r_height), 2)  # 最大轮廓的百分比
            if areaMaxContour is not None:
                rect = cv2.minAreaRect(areaMaxContour)  # 矩形框选
                box = np.int0(cv2.boxPoints(rect))      # 点的坐标
                if img_debug:
                    cv2.drawContours(handling, [box], 0, (153, 200, 0), 2)  # 将最小外接矩形画在图上

            if img_debug:
                cv2.putText(handling, 'area: ' + str(percent) + '%', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('handling', handling)  # 显示图像

                #cv2.imshow('frame_door_yellow', frame_door_yellow)  # 显示图像
                #cv2.imshow('frame_door_black', frame_door_black)    # 显示图像
                

                k = cv2.waitKey(10)
                if k == 27:
                    cv2.destroyWindow('open_after_closed')
                    cv2.destroyWindow('handling')
                    break
                elif k == ord('s'):
                    print("save picture123")
                    cv2.imwrite("picture123.jpg",org_img_copy) #保存图片



            # 根据比例得到是否前进的信息
            if percent > 5:    #检测到横杆
                print(percent,"%")
                print("有障碍 等待 contours len：",len(contours))
                action_append("Stand")
                end_door_flag = 1
                time.sleep(3)
            else:
                if end_door_flag == 0:
                    print(percent)
                    print("暂未发现横杆 等待检测")
                    action_append("Stand")

                elif end_door_flag == 1 and percent <1 :
                    print(percent)
                    # print("3894L 执行3步")
                    # action_append("forwardSlow0403")
                    # action_append("forwardSlow0403")
                    # action_append("forwardSlow0403")

                    print("3899L 执行快走555")
                    action_append("fastForward05")
                    action_append("Stand")
                    step = 1

                else:
                    print(percent,"%")
                    print("有障碍 等待 contours len：",len(contours))
                    action_append("Stand")
                    time.sleep(3)
                
        elif step == 1:  
            break



#################################################台阶##########################################
def floor():
    global org_img, state, state_sel, step, reset, skip, debug
    global camera_out
    state_sel = 'floor'

    if state_sel == 'floor':  # 初始化
        print("/-/-/-/-/-/-/-/-/-进入floor")
        step = 0

    r_w = chest_r_width
    r_h = chest_r_height

    top_angle = 0
    T_B_angle = 0
    topcenter_x = 0.5 * r_w
    topcenter_y = 0
    bottomcenter_x = 0.5 * r_w
    bottomcenter_y = 0

    topcenter_x_setl=230
    topcenter_x_setr=250
    #topcenter_y_setu=280
    #bottomcenter_x_setl=0
    #bottomcenter_x_setr=0
    while state_sel == 'floor':
        # chest
        if True:  # 上下边沿
            t1 = cv2.getTickCount()
            Corg_img = ChestOrg_img.copy()
            Corg_img = np.rot90(Corg_img)
            OrgFrame = Corg_img.copy()

            # 初始化 bottom_right  bottom_left
            bottom_right = (480, 0)
            bottom_left = (0, 0)
            top_right = (480, 0)  # 右上角点坐标
            top_left = (0, 0)  # 左上角点坐标

            frame = cv2.resize(OrgFrame, (chest_r_width, chest_r_height), interpolation=cv2.INTER_LINEAR)
            frame_copy = frame.copy()
            # 获取图像中心点坐标x, y
            center = []
            # 开始处理图像
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
            if step == 0:
                Imask = cv2.inRange(hsv, color_range['blue_floor'][0],
                                    color_range['blue_floor'][1])  # 对原图像和掩模(颜色的字典)进行位运算
            elif step == 1:
                Imask = cv2.inRange(hsv, color_range['blue_floor'][0], color_range['blue_floor'][1])
            elif step == 2:
                Imask = cv2.inRange(hsv, color_range['green_floor'][0], color_range['green_floor'][1])
            elif step == 3:
                Imask1 = cv2.inRange(hsv, color_range['red_floor1'][0], color_range['red_floor1'][1])
                Imask2 = cv2.inRange(hsv, color_range['red_floor2'][0], color_range['red_floor2'][1])
                Imask =cv2.bitwise_or(Imask1,Imask2)
            elif step == 4:
                Imask = cv2.inRange(hsv, color_range['green_floor'][0], color_range['green_floor'][1])
            elif step == 5:
                Imask = cv2.inRange(hsv, color_range['blue_floor'][0], color_range['blue_floor'][1])
            elif step == 6 or step == 6.1 or step ==7:
                frame_1 = cv2.inRange(hsv, color_range['red_XP1'][0], color_range['red_XP1'][1])  # 对原图像和掩模(颜色的字典)进行位运算
                frame_2 = cv2.inRange(hsv, color_range['red_XP2'][0], color_range['red_XP2'][1])
                Imask = cv2.bitwise_or(frame_1, frame_2)
                # Imask = cv2.inRange(hsv, color_range['blue_floor'][0], color_range['blue_floor'][1])
            else:
                print("no color")

            # opened = cv2.morphologyEx(Imask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算 去噪点
            # Imask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算 封闭连接

            # Imask = cv2.erode(Imask, None, iterations=2)
            Imask = cv2.dilate(Imask, np.ones((3, 3), np.uint8), iterations=2)

            _, cnts, hierarchy = cv2.findContours(Imask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓

            cnt_sum, area_max = getAreaMaxContour1(cnts)  # 找出最大轮廓
            C_percent = round(area_max * 100 / (r_w * r_h), 2)  # 最大轮廓百分比
            cv2.drawContours(frame, cnt_sum, -1, (255, 0, 255), 3)

            if cnt_sum is not None:
                see = True
                rect = cv2.minAreaRect(cnt_sum)  # 最小外接矩形
                box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点
                bottom_right = cnt_sum[0][0]  # 右下角点坐标
                bottom_left = cnt_sum[0][0]  # 左下角点坐标
                top_right = cnt_sum[0][0]  # 右上角点坐标
                top_left = cnt_sum[0][0]  # 左上角点坐标
                # #杨巍改的算矩形四个顶点位置的方法
                # left_x=cnt_sum[0][0][0]
                # right_x=cnt_sum[0][0][0]
                # up_y=cnt_sum[0][0][1]
                # down_y=cnt_sum[0][0][1]
                for c in cnt_sum:
                    if c[0][0] + 1 * (r_h - c[0][1]) < bottom_left[0] + 1 * (r_h - bottom_left[1]):
                        bottom_left = c[0]
                    if c[0][0] + 1 * c[0][1] > bottom_right[0] + 1 * bottom_right[1]:
                        bottom_right = c[0]

                    if c[0][0] + 3 * c[0][1] < top_left[0] + 3 * top_left[1]:
                        top_left = c[0]
                    if (r_w - c[0][0]) + 3 * c[0][1] < (r_w - top_right[0]) + 3 * top_right[1]:
                        top_right = c[0]
                    # if c[0][0] < left_x:
                    #     left_point = c[0]
                    # if c[0][0] > right_x:
                    #     right_point = c[0]
                    # if c[0][1]<up_y:
                    #     up_point = c[0]
                    # if c[0][1]>down_y:
                    #     down_point = c[0]


                    # if debug:
                    #     handling = ChestOrg_img.copy()
                    #     cv2.circle(handling, (c[0][0], c[0][1]), 5, [0, 255, 0], 2)
                    #     cv2.circle(handling, (bottom_left[0], bottom_left[1]), 5, [255, 255, 0], 2)
                    #     cv2.circle(handling, (bottom_right[0], bottom_right[1]), 5, [255, 0, 255], 2)
                    #     cv2.imshow('handling', handling)  # 显示图像
                    #     cv2.waitKey(2)

                bottomcenter_x = (bottom_left[0] + bottom_right[0]) / 2  # 得到bottom中心坐标
                bottomcenter_y = (bottom_left[1] + bottom_right[1]) / 2

                topcenter_x = (top_right[0] + top_left[0]) / 2  # 得到top中心坐标
                topcenter_y = (top_left[1] + top_right[1]) / 2

                bottom_angle = -math.atan(
                    (bottom_right[1] - bottom_left[1]) / (bottom_right[0] - bottom_left[0])) * 180.0 / math.pi
                top_angle = -math.atan((top_right[1] - top_left[1]) / (top_right[0] - top_left[0])) * 180.0 / math.pi
                if math.fabs(topcenter_x - bottomcenter_x) <= 1:  # 得到连线的角度
                    T_B_angle = 90
                else:
                    T_B_angle = - math.atan(
                        (topcenter_y - bottomcenter_y) / (topcenter_x - bottomcenter_x)) * 180.0 / math.pi

                if img_debug:
                    cv2.drawContours(frame_copy, [box], 0, (0, 255, 0), 2)  # 将大矩形画在图上
                    cv2.line(frame_copy, (bottom_left[0], bottom_left[1]), (bottom_right[0], bottom_right[1]),
                             (255, 255, 0), thickness=2)
                    cv2.line(frame_copy, (top_left[0], top_left[1]), (top_right[0], top_right[1]), (255, 255, 0),
                             thickness=2)
                    cv2.line(frame_copy, (int(bottomcenter_x), int(bottomcenter_y)),
                             (int(topcenter_x), int(topcenter_y)), (255, 255, 255), thickness=2)  # T_B_line

                    cv2.putText(frame_copy, "bottom_angle:" + str(bottom_angle), (30, 450), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
                    cv2.putText(frame_copy, "top_angle:" + str(top_angle), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 0, 0), 2)
                    cv2.putText(frame_copy, "T_B_angle:" + str(T_B_angle), (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 0, 255), 2)

                    cv2.putText(frame_copy, "bottomcenter_x:" + str(bottomcenter_x), (30, 480),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
                    cv2.putText(frame_copy, "y:" + str(int(bottomcenter_y)), (300, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 0, 0), 2)  # (0, 0, 255)BGR

                    cv2.putText(frame_copy, "topcenter_x:" + str(topcenter_x), (30, 180), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
                    cv2.putText(frame_copy, "topcenter_y:" + str(int(topcenter_y)), (230, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR

                    cv2.putText(frame_copy, 'C_percent:' + str(C_percent) + '%', (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 0), 2)
                    cv2.putText(frame_copy, "step:" + str(step), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),
                                2)  # (0, 0, 255)BGR

                    cv2.circle(frame_copy, (int(topcenter_x), int(topcenter_y)), 5, [255, 0, 255], 2)
                    cv2.circle(frame_copy, (int(bottomcenter_x), int(bottomcenter_y)), 5, [255, 0, 255], 2)
                    cv2.circle(frame_copy, (top_right[0], top_right[1]), 5, [0, 255, 255], 2)
                    cv2.circle(frame_copy, (top_left[0], top_left[1]), 5, [0, 255, 255], 2)
                    cv2.circle(frame_copy, (bottom_right[0], bottom_right[1]), 5, [0, 255, 255], 2)
                    cv2.circle(frame_copy, (bottom_left[0], bottom_left[1]), 5, [0, 255, 255], 2)
                    cv2.imshow('Chest_Camera', frame_copy)  # 显示图像
                    cv2.imshow('chest_red_mask', Imask)
                    cv2.waitKey(100)
                # 决策执行动作
                angle_ok_flag = False

                if step == 0:  # 前进依据chest 调整大致位置，方向  看底边线调整角度

                    if bottomcenter_y < 300:
                        if bottom_angle > 3:  # 需要左转
                            if bottom_angle > 6:
                                print("4085L 大左转一下  turn001L ", bottom_angle)
                                action_append("turn001L")
                            else:
                                print("4088L bottom_angle > 3 需要小左转 turn001L ", bottom_angle)
                                action_append("turn001L")
                        elif bottom_angle < -3:  # 需要右转
                            if bottom_angle < -6:
                                print("4092L 右da旋转  turn001R < -6 ", Head_L_R_angle)
                                action_append("turn001R")
                            else:
                                print("4095L bottom_angle < -3 需要小右转 turn001R ", bottom_angle)
                                action_append("turn001R")
                        elif -3 <= bottom_angle <= 3:  # 角度正确
                            print("4098L 角度合适")

                            if topcenter_x > topcenter_x_setr or topcenter_x < topcenter_x_setl:
                                if topcenter_x > topcenter_x_setr:
                                    print("微微右移,", topcenter_x)
                                    action_append("Right3move")
                                elif topcenter_x < topcenter_x_setl:
                                    print("微微左移,", topcenter_x)
                                    action_append("Left3move")

                            else:
                                print("位置合适")
                                print("快步走,bottomcenter_y", bottomcenter_y)
                                action_append("fastForward04")

                    elif bottomcenter_y < 380:
                        if bottom_angle > 3:  # 需要左转
                            if bottom_angle > 6:
                                print("4116L 大左转一下  turn001L ", bottom_angle)
                                action_append("turn001L")
                            else:
                                print("4119L bottom_angle > 3 需要小左转 turn001L ", bottom_angle)
                                action_append("turn001L")
                        elif bottom_angle < -3:  # 需要右转
                            if bottom_angle < -6:
                                print("4123L 右da旋转  turn001R < -6 ", Head_L_R_angle)
                                action_append("turn001R")
                            else:
                                print("4126L bottom_angle < -3 需要小右转 turn001R ", bottom_angle)
                                action_append("turn001R")
                        elif -3 <= bottom_angle <= 3:  # 角度正确
                            print("4129L 角度合适")
                            angle_ok_flag = True

                        if angle_ok_flag:
                            if topcenter_x > topcenter_x_setr or topcenter_x < topcenter_x_setl:
                                if topcenter_x > topcenter_x_setr:
                                    print("微微右移,", topcenter_x)
                                    action_append("Right02move")
                                elif topcenter_x < topcenter_x_setl:
                                    print("微微左移,", topcenter_x)
                                    action_append("Left02move")
                            else:
                                print("4141L 继续前行 forwardSlow0403", bottomcenter_y)
                                action_append("forwardSlow0403")

                    elif 380 <= bottomcenter_y < 430:
                        if bottom_angle > 3:  # 需要左转
                            if bottom_angle > 6:
                                print("4147L 大左转一下  turn001L ", bottom_angle)
                                action_append("turn001L")
                            else:
                                print("4150L bottom_angle > 3 需要小左转 turn001L ", bottom_angle)
                                action_append("turn001L")
                        elif bottom_angle < -3:  # 需要右转
                            if bottom_angle < -6:
                                print("4154L 右da旋转  turn001R < -6 ", Head_L_R_angle)
                                action_append("turn001R")
                            else:
                                print("4157L bottom_angle < -3 需要小右转 turn001R ", bottom_angle)
                                action_append("turn001R")
                        elif -3 <= bottom_angle <= 3:  # 角度正确
                            print("4160L 角度合适")
                            angle_ok_flag = True

                        if angle_ok_flag:
                            if topcenter_x > topcenter_x_setr or topcenter_x < topcenter_x_setl:
                                if topcenter_x > topcenter_x_setr:
                                    print("微微右移,", topcenter_x)
                                    action_append("Right02move")
                                elif topcenter_x < topcenter_x_setl:
                                    print("微微左移,", topcenter_x)
                                    action_append("Left02move")
                            else:
                                print("4172L 变小步继续前行 Forwalk00", bottomcenter_y)
                                action_append("Forwalk01")

                    elif 430 <= bottomcenter_y <= 540:
                        if bottom_angle > 3:  # 需要左转
                            if bottom_angle > 6:
                                print("4178L 大左转一下  turn001L ", bottom_angle)
                                action_append("turn001L")
                            else:
                                print("4181L bottom_angle > 3 需要小左转 turn001L ", bottom_angle)
                                action_append("turn001L")
                        elif bottom_angle < -3:  # 需要右转
                            if bottom_angle < -6:
                                print("4185L 右da旋转  turn001R < -6 ", bottom_angle)
                                action_append("turn001R")
                            else:
                                print("4188L bottom_angle < -3 需要小右转 turn001R ", bottom_angle)
                                action_append("turn001R")
                        elif -3 <= bottom_angle <= 3:  # 角度正确
                            print("4191L 角度合适")
                            angle_ok_flag = True

                        if angle_ok_flag:
                            if topcenter_x > topcenter_x_setr or topcenter_x < topcenter_x_setl:
                                if topcenter_x > topcenter_x_setr:
                                    print("微微右移,", topcenter_x)
                                    action_append("Right02move")
                                elif topcenter_x < topcenter_x_setl:
                                    print("微微左移,", topcenter_x)
                                    action_append("Left02move")
                            else:
                                print("4203L 到达上台阶边沿，变前挪动 Forwalk00 bottomcenter_x:", bottomcenter_x)
                                action_append("Forwalk00")


                    elif bottomcenter_y > 540:
                        print("然后开始第二步------上第一节台阶")
                        step = 1
                        angle_ok_flag = False
                    else:
                        print("error 前进 C_percent:", C_percent)
                        print("bottomcenter_y:", bottomcenter_y)

                elif step == 1:  # 看中线调整角度上台阶----第一阶

                    if top_angle < -2:  # 右转
                        print("4236L 右转 turn001R top_angle:", top_angle)
                        action_append("turn001R")
                        time.sleep(0.5)  # timefftest
                    elif top_angle > 2:  # 左转
                        print("4240L 左转 turn001L top_angle:", top_angle)
                        action_append("turn001L")
                        time.sleep(0.5)  # timefftest
                    elif -2 <= top_angle <= 2:
                        print("前走一小步")
                        action_append("Forwalk00")
                        time.sleep(0.5)
                        print("4247L 上台阶 上台阶 UpBridge")
                        action_append("UpBridge16")
                        print("————————————————————————开始上第二节台阶")
                        time.sleep(0.5)

                        step = 2


                elif step == 2:  # 看中线调整角度上台阶----第二阶
                    # if 0 < T_B_angle < 85:  # 右转
                    #     print("4257L 右转 turn001R T_B_angle:",T_B_angle)
                    #     action_append("turn001R")
                    #     time.sleep(0.5)   # timefftest
                    # elif -85 < T_B_angle < 0:  # 左转
                    #     print("4261L 左转 turn001L T_B_angle:",T_B_angle)
                    #     action_append("turn001L")
                    #     time.sleep(0.5)   # timefftest
                    # elif T_B_angle <= -85 or T_B_angle >= 85:
                    #     print("4265L 上台阶 上台阶 UpBridge")
                    #     action_append("UpBridge")

                    #     print("————————————————————————开始上第三节台阶")
                    #     time.sleep(0.5)

                    #     step = 3

                    if top_angle < -2:  # 右转
                        print("4274L 右转 turn001R top_angle:", top_angle)
                        action_append("turn001R")
                        time.sleep(0.5)  # timefftest
                    elif top_angle > 2:  # 左转
                        print("4278L 左转 turn001L top_angle:", top_angle)
                        action_append("turn001L")
                        time.sleep(0.5)  # timefftest
                    elif -2 <= top_angle <= 2:
                        print("4282L 上台阶 上台阶 UpBridge")
                        action_append("UpBridge16")

                        print("————————————————————————开始上第三节台阶")
                        time.sleep(0.5)

                        step = 3

                elif step == 3:  # 看中线调整角度上台阶----第三阶
                    # if 0 < T_B_angle < 85:  # 右转
                    #     print("711L 右转 turn001R T_B_angle:",T_B_angle)
                    #     action_append("turn001R")
                    #     time.sleep(0.5)   # timefftest
                    # elif -85 < T_B_angle < 0:  # 左转
                    #     print("715L 左转 turn001L T_B_angle:",T_B_angle)
                    #     action_append("turn001L")
                    #     time.sleep(0.5)   # timefftest
                    # elif T_B_angle <= -85 or T_B_angle >= 85:
                    #     print("719L 上台阶 上台阶 UpBridge")
                    #     action_append("UpBridge")

                    #     print("————————————————————————上台阶完毕，开始下台阶")
                    #     print("————————————————————————开始下第一节台阶")
                    #     time.sleep(0.5)

                    #     step = 4

                    if top_angle < -2:  # 右转
                        print("4310L 右转 turn001R top_angle:", top_angle)
                        action_append("turn001R")
                        time.sleep(0.5)  # timefftest
                    elif top_angle > 2:  # 左转
                        print("4314L 左转 turn001L top_angle:", top_angle)
                        action_append("turn001L")
                        time.sleep(0.5)  # timefftest
                    elif -2 <= top_angle <= 2:
                        print("4318L 上台阶 上台阶 UpBridge")
                        action_append("UpBridge16")

                        print("————————————————————————上台阶完毕，开始下台阶")
                        print("————————————————————————开始下第一节台阶")
                        time.sleep(0.5)

                        step = 4

                        # print("4328L 上台阶后，前进一步")
                        # action_append("Forwalk01")

                        # print("按键继续。。。")
                        # cv2.waitKey(0)

                elif step == 4:  # 调整角度下台阶----第三阶
                    time.sleep(0.5)
                    if top_angle > 2:  # 需要左转
                        print("4337 top_angle > 2 需要小左转 ")
                        action_append("turn001L")
                    elif top_angle < -2:  # 需要右转
                        print("4340 top_angle < -2 需要小右转 ")
                        action_append("turn001R")
                    elif -2 <= top_angle <= 2:  # 角度正确
                        print("角度合适")
                        if topcenter_y < 380:
                            print("微微前挪")
                            action_append("Forwalk00")
                        elif topcenter_y > 380:
                            print("4348L 下台阶 下台阶 DownBridge topcenter_y:", topcenter_y)
                            action_append("DownBridge")
                            print("————————————————————————开始下第二节台阶")
                            time.sleep(0.5)
                            step = 5

                elif step == 5:  # 调整角度下台阶----第二阶
                    time.sleep(0.5)
                    if top_angle > 2:  # 需要左转
                        print("4357 top_angle > 2 需要小左转 ")
                        action_append("turn001L")
                    elif top_angle < -2:  # 需要右转
                        print("4360 top_angle < -2 需要小右转 ")
                        action_append("turn001R")
                    elif -2 <= top_angle <= 2:  # 角度正确
                        print("角度合适")
                        if topcenter_y < 380:
                            print("微微前挪")
                            action_append("Forwalk00")
                        elif topcenter_y > 380:
                            print("4368L 下台阶 下台阶 DownBridge topcenter_y:", topcenter_y)
                            action_append("DownBridge")
                            print("————————————————————————开始下第二节台阶")
                            time.sleep(0.5)
                            step = 6

                elif step == 6:  # 调整角度下斜坡----第三阶
                    time.sleep(0.5)

                    if top_angle > 2:  # 需要左转
                        print("4377 top_angle > 2 需要小左转 y:", topcenter_y)
                        action_append("turn001L")
                    elif top_angle < -2:  # 需要右转
                        print("4380 top_angle < -2 需要小右转 y:", topcenter_y)
                        action_append("turn001R")
                    elif -2 <= top_angle <= 2:  # 角度正确

                        print("角度合适")
                        print("topcenter_x=", topcenter_x)
                        if topcenter_x > topcenter_x_setr or topcenter_x < topcenter_x_setl:
                            if topcenter_x > topcenter_x_setr:
                                print("微微右移", topcenter_x)
                                action_append("Right02move")
                            elif topcenter_x < topcenter_x_setl:
                                print("微微左移", topcenter_x)
                                action_append("Left02move")

                        # print("微微前挪 y:",topcenter_y)
                        # action_append("Forwalk00")
                        else:
                            print("位置合适")
                            print("下斜坡,step=6.1，后倾预备")
                            action_append("Forwalk00")
                            action_append("Stand")
                            action_append("actBeforeXP")
                            step =6.1

                elif step == 6.1:
                    if area_max>20:
                        print("面积大于20则继续下坡走，XPforwalkSlow")
                        action_append("XPforwalkSlow")
                    else:
                        print("再走几步就XP结束了")

                        action_append("XPforwalkSlow")
                        action_append("XPforwalkSlow")
                        action_append("XPforwalkSlow")
                        action_append("XPforwalkSlow")
                        step = 7

                elif step == 7:  # 完成

                    print("899L 完成floor")
                    action_append("Forwalk00")
                    action_append("Forwalk00")
                    action_append("Forwalk00")
                    action_append("Forwalk00")
                    action_append("Forwalk00")
                    break
            elif step == 6.1:
                print("找不到轮廓了，证明下坡结束了")
                action_append("XPforwalkSlow")
                action_append("XPforwalkSlow")
                action_append("XPforwalkSlow")
                action_append("XPforwalkSlow")
                step=7
            elif step == 7:
                print("899L 完成floor")
                action_append("Forwalk00")
                action_append("Forwalk00")
                action_append("Forwalk00")
                action_append("Forwalk00")
                action_append("Forwalk00")
                break
            else:
                print("未找到第一届蓝色台阶")
                action_append("Forwalk00")





###################### 起            点-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
def start_door():
    global ChestOrg_img, state, state_sel, step, img_debug,door_flag
    start_door_flag = 0
    state_sel = 'start_door'
    state = 1
    if state == 1:  # 初始化
        print("/-/-/-/-/-/-/-/-/-进入door")
        step = 0
    else:
        pass

    while state == 1 :

        if step == 0: #判断门是否抬起
            if ChestOrg_img is None:
                continue
            
            org_img_copy = ChestOrg_img.copy()
            org_img_copy = np.rot90(org_img_copy)
            handling = org_img_copy.copy()           

            border = cv2.copyMakeBorder(handling, 12, 12, 16, 16, borderType=cv2.BORDER_CONSTANT,value=(255, 255, 255))     # 扩展白边，防止边界无法识别
            handling = cv2.resize(border, (chest_r_width, chest_r_height), interpolation=cv2.INTER_CUBIC)                   # 将图片缩放
            frame_gauss = cv2.GaussianBlur(handling, (21, 21), 0)       # 高斯模糊
            frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)    # 将图片转换到HSV空间
            
            frame_door_yellow = cv2.inRange(frame_hsv, color_range['yellow_door'][0], color_range['yellow_door'][1])    # 对原图像和掩模(颜色的字典)进行位运算
            frame_door_black = cv2.inRange(frame_hsv, color_range['black_door'][0], color_range['black_door'][1])       # 对原图像和掩模(颜色的字典)进行位运算


            frame_door = cv2.add(frame_door_yellow, frame_door_black)    
            open_pic = cv2.morphologyEx(frame_door, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))      # 开运算 去噪点
            closed_pic = cv2.morphologyEx(open_pic, cv2.MORPH_CLOSE, np.ones((50, 50), np.uint8))   # 闭运算 封闭连接        

            (image, contours, hierarchy) = cv2.findContours(closed_pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓
            areaMaxContour, area_max = getAreaMaxContour1(contours)  # 找出最大轮廓
            percent = round(100 * area_max / (chest_r_width * chest_r_height), 2)  # 最大轮廓的百分比
            if areaMaxContour is not None:
                rect = cv2.minAreaRect(areaMaxContour)  # 矩形框选
                box = np.int0(cv2.boxPoints(rect))      # 点的坐标
                if img_debug:
                    cv2.drawContours(handling, [box], 0, (153, 200, 0), 2)  # 将最小外接矩形画在图上

            if img_debug:
                cv2.putText(handling, 'area: ' + str(percent) + '%', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('handling', handling)  # 显示图像

                #cv2.imshow('frame_door_yellow', frame_door_yellow)  # 显示图像
                #cv2.imshow('frame_door_black', frame_door_black)    # 显示图像
                

                k = cv2.waitKey(10)
                if k == 27:
                    cv2.destroyWindow('open_after_closed')
                    cv2.destroyWindow('handling')
                    break
                elif k == ord('s'):
                    print("save picture123")
                    cv2.imwrite("picture123.jpg",org_img_copy) #保存图片



            # 根据比例得到是否前进的信息
            if percent > 5:    #检测到横杆
                print(percent,"%")
                print("有障碍 等待 contours len：",len(contours))
                action_append("Stand")
                start_door_flag = 1
                time.sleep(3)
            else:
                if start_door_flag == 0:
                    print(percent)
                    print("暂未发现横杆 等待检测")
                    action_append("Stand")

                elif start_door_flag == 1 and percent <1:
                    print(percent)
                    # print("3894L 执行3步")
                    # action_append("forwardSlow0403")
                    # action_append("forwardSlow0403")
                    # action_append("forwardSlow0403")

                    print("3899L 执行快走555")
                    action_append("fastForward03")
                    action_append("Stand")
                    step = 1

                else:
                    print(percent,"%")
                    print("有障碍 等待 contours len：",len(contours))
                    action_append("Stand")
                    time.sleep(3)
                
        elif step == 1:  
            break
            
def get_img():
    global ChestOrg_img, HeadOrg_img, HeadOrg_img, chest_ret
    global ret
    global cap_chest
    while True:
        if 1:
        # if not img_debug:
            if cap_chest.isOpened():

                chest_ret, ChestOrg_img = cap_chest.read()
                ret, HeadOrg_img = cap_head.read()
                if (chest_ret == False) or (ret == False):
                    print("ret fail ------------------")
                if HeadOrg_img is None:
                    print("HeadOrg_img error")
                if ChestOrg_img is None:
                    print("ChestOrg_img error")

            else:
                time.sleep(0.01)
                ret = True
                print("4568L pic  error ")

        else:
            ChestOrg_img = cv2.imread("../img_dbg/1.jpg")

# 读取图像线程

th1 = threading.Thread(target=get_img)
th1.setDaemon(True)
th1.start()


def move_action():
    global org_img
    global step, level
    global golf_angle_hole
    global golf_angle_ball, golf_angle
    global golf_dis, golf_dis_y
    global golf_angle_flag, golf_dis_flag
    global golf_angle_start, golf_dis_start
    global golf_ok
    global golf_hole, golf_ball

    if real_test:
        CMDcontrol.CMD_transfer()

# 动作执行线程
th2 = threading.Thread(target=move_action)
th2.setDaemon(True)
th2.start()


if __name__ == '__main__':
    if real_test:
        while len(CMDcontrol.action_list) > 0:
            print("等待启动")
            time.sleep(1)
        action_append("HeadTurnMM") #yw:headturnmm 是把头部转到100位置，应该是归零（归位）

    while True:
        if ChestOrg_img is not None and chest_ret:
            k = cv2.waitKey(10)#yw：换行符
            if k == 27:#yw：ESC键
                cv2.destroyWindow('camera_test')
                break
            
            if single_debug:#yw：每执行一次动作停顿一下
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            print("start door START")
            t1 = cv2.getTickCount()#yw：这个函数返回CPU的时间（但是得到的是周期数，要换成秒的话需要除以频率），取两次时间就可以得到时间差。t2-t1
            f = cv2.getTickFrequency()#yw：这个函数返回CPU时间的频率。
            start_door()
            t2 = cv2.getTickCount()
            print("start door Execution time: {}".format((t2-t1)/f))
            if single_debug:
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()


            while True:
                  
                i = recognize()
                if i == 5 or i == 9:
                    flag = 1
                    print("Single log bridge START")
                    t1 = cv2.getTickCount()
                    f = cv2.getTickFrequency()
                    if i == 5:
                        Greenbridge('green_bridge')
                    elif i ==9:
                        Greenbridge('blue_bridge')
                    t2 = cv2.getTickCount()
                    print("Single log bridge Execution time: {}".format((t2-t1)/f))
                    if single_debug:
                        print("Press any key to continue...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    cv2.destroyAllWindows()   
                    break

                elif i == 1 or i == 10:
                    flag = 2
                    print("Through Pit START")
                    t1 = cv2.getTickCount()
                    f = cv2.getTickFrequency()
                    if i == 1:
                        hole_edge('green_hole_chest')
                    elif i == 10:
                        hole_edge('blue_hole_chest')
                    t2 = cv2.getTickCount()
                    print("Through Pit Execution time: {}".format((t2-t1)/f))
                    if single_debug:
                        print("Press any key to continue...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    cv2.destroyAllWindows()
                    break

                elif i == 0:
                    print("Error! 未识别到有效关卡")
                    action_append("Forwalk00")
                    # cv2.imshow(HeadOrg_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    continue#yw：这里便是有无限循环的可能了，如果一直识别不到关卡，应该做一些其他动作来跳出该循环。可在进入该循环时设一个t0，t如果大于比如10s那么跳出循环去做点什么。

            # yw：下面这几段代码，直接调用了相应的关卡函数，因为在关卡内部有识别该关卡的方法。
            print("Through obstacle START")#yw：过雷阵
            t1 = cv2.getTickCount()
            f = cv2.getTickFrequency()
            obstacle()
            t2 = cv2.getTickCount()
            print("Through obstacle Execution time: {}".format((t2-t1)/f))
            if single_debug:
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            
            print("Through baffle START")#yw：过挡板
            t1 = cv2.getTickCount()
            f = cv2.getTickFrequency()
            baffle()
            t2 = cv2.getTickCount()
            print("Through baffle Execution time: {}".format((t2-t1)/f))
            if single_debug:
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            
            print("Into Door START")
            t1 = cv2.getTickCount()
            f = cv2.getTickFrequency()
            into_the_door()
            t2 = cv2.getTickCount()
            print("Into Door START Execution time: {}".format((t2-t1)/f))
            if single_debug:
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            if flag == 2:
                area_green = area_calculate('green_bridge')
                area_blue = area_calculate('blue_bridge')
                if(area_green > area_blue):
                    print("Single log green bridge START")
                    Greenbridge('green_bridge')
                elif(area_blue > area_green):
                    print("Single log blue bridge START")
                    Greenbridge('blue_bridge')

            elif flag == 1:
                area_green = area_calculate('green_hole_chest')
                area_blue = area_calculate('blue_hole_chest')
                if(area_green > area_blue):
                    print("Single log green hole START")
                    hole_edge('green_hole_chest')
                elif(area_blue > area_green):
                    print("Single log blue bridge START")
                    hole_edge('blue_hole_chest')
            
            print("Kick ball START")
            t1 = cv2.getTickCount()
            f = cv2.getTickFrequency()
            kick_ball()
            t2 = cv2.getTickCount()
            print("Kick ball Execution time: {}".format((t2-t1)/f))
            if single_debug:
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()   
            
            print("Floor START")
            t1 = cv2.getTickCount()
            f = cv2.getTickFrequency()
            floor()
            t2 = cv2.getTickCount()
            print("Floor Execution time: {}".format((t2-t1)/f))
            if single_debug:
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            print("End Door START")
            t1 = cv2.getTickCount()
            f = cv2.getTickFrequency()
            end_door()
            t2 = cv2.getTickCount()
            print("End Door Execution time: {}".format((t2-t1)/f))
            if single_debug:
                print("Press any key to continue...")
                cv2.waitKey(0)  
                cv2.destroyAllWindows()      
            
            print("End Door START")
            t1 = cv2.getTickCount()
            f = cv2.getTickFrequency()
            end_door()
            t2 = cv2.getTickCount()
            print("End Door Execution time: {}".format((t2-t1)/f))
            if single_debug:
                print("Press any key to continue...")
                cv2.waitKey(0)  
                cv2.destroyAllWindows()
            while (1):
                print("结束")
                time.sleep(10000)
            
            
        else:
            print('image is empty chest_ret:', chest_ret)
            time.sleep(0.01)
            cv2.destroyAllWindows()
