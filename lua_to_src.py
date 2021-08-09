#!/usr/bin/env python3
# coding:utf-8
"""比赛方居然没有一个lua转src的程序，那我自己写一个-----杨巍“”“
”“”使用时将你想转换的function写到lua.txt文件里，放在该py
    文件同目录下，然后点击运行便开始转换了"""
import re
import time
fi = open("lua.txt","r",encoding="utf8")     # f就是文件对象，一般情况下，只设置这3个参数
fo = open("srcfile"+str(time.ctime().replace(":","-"))+".src","w+")
lines=fi.readlines()
for line in lines:
    line=line.replace(" ","")
    if "MOTOsetspeed" in line:
        list=re.split('[,()]',line)
        print(list)
        fo.writelines("\n"+"SPEED "+str(list[1])+"\n")
    elif "MOTOrigid16" in line:
        list=re.split('[,()]',line)
        print(list)
        if len(list)==18:
            list.append('0')
            list.append('0')
            list.append('0')
        fo.writelines("RIGIDA,"+str(list[8])+","+str(list[7])+","+
                      str(list[6])+","+str(list[5])+","+str(list[4])+"\n")
        fo.writelines("RIGIDB," + str(list[16]) + "," + str(list[15]) + "," +
                      str(list[14]) + "," + str(list[13]) + "," + str(list[12])+ "\n")
        fo.writelines("RIGIDC," + str(list[3]) + "," + str(list[2]) + "," +
                      str(list[1]) + "\n")
        fo.writelines("RIGIDD," + str(list[11]) + "," + str(list[10]) + "," +
                      str(list[9]) + "\n")
        fo.writelines("RIGIDE," + str(list[19]) + "," + str(list[18]) + "," +
                      str(list[17]) + "\n")
        fo.writelines("RIGEND"+"\n")
    elif "MOTOmove19" in line:
        list = re.split('[,()]', line)
        print(list)
        fo.writelines("MOTORA," + str(list[8]) + "," + str(200-eval(list[7])) + "," +
                      str(200-eval(list[6])) + "," + str(list[5]) + "," + str(200-eval(list[4])) + "\n")
        fo.writelines("MOTORB," + str(200-eval(list[16])) + "," + str(list[15]) + "," +
                      str(list[14]) + "," + str(200-eval(list[13])) + "," + str(list[12]) + "\n")
        fo.writelines("MOTORC," + str(list[3]) + "," + str(list[2]) + "," +
                      str(list[1]) + "\n")
        fo.writelines("MOTORD," + str(200-eval(list[11])) + "," + str(200-eval(list[10])) + "," +
                      str(200-eval(list[9])) + "\n")

        fo.writelines("MOTORE," + str(100 if (list[19]=='00' or list[19]=='0') else list[19]) + "," + str(list[18]) + "," +
                      str(list[17]) + "\n")
    elif "wait" in line:
        list = re.split('[,()]', line)
        print(list)
        fo.writelines("WAIT "+str(list[1])+"\n")
    elif "DelayMs"in line:
        list = re.split('[,()]', line)
        print(list)
        fo.writelines("DELAY " + str(list[1]) + "\n")
    else:
        continue
fo.close()
