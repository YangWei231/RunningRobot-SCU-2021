#!/usr/bin/env python3
# coding:utf-8
"""比赛方居然没有一个lua转src的程序，那我自己写一个-----杨巍“”“
”“”使用时将你想转换的function写到lua.txt文件里，放在该py
    文件同目录下，然后点击运行便开始转换了"""
import re
fi = open("lua.txt","r",encoding="utf8")     # f就是文件对象，一般情况下，只设置这3个参数
fo = open("srcfile.txt","w+")
lines=fi.readlines()
for line in lines:
    line=line.replace(" ","")
    if "MOTOsetspeed" in line:

        list=re.split('[,()]',line)
        fo.writelines("\n"+"SPEED "+str(list[1])+"\n")
    elif "MOTOrigid16" in line:
        list=re.split('[,()]',line)
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
        fo.writelines("MOTORA," + str(list[8]) + "," + str(list[7]) + "," +
                      str(list[6]) + "," + str(list[5]) + "," + str(list[4]) + "\n")
        fo.writelines("MOTORB," + str(list[16]) + "," + str(list[15]) + "," +
                      str(list[14]) + "," + str(list[13]) + "," + str(list[12]) + "\n")
        fo.writelines("MOTORC," + str(list[3]) + "," + str(list[2]) + "," +
                      str(list[1]) + "\n")
        fo.writelines("MOTORD," + str(list[11]) + "," + str(list[10]) + "," +
                      str(list[9]) + "\n")
        fo.writelines("MOTORE," + str(list[19]) + "," + str(list[18]) + "," +
                      str(list[17]) + "\n")
    elif "wait" in line:
        list = re.split('[,()]', line)
        fo.writelines("WAIT "+str(list[1])+"\n")
    elif "DelayMs"in line:
        list = re.split('[,()]', line)
        fo.writelines("DELAY " + str(list[1]) + "\n")
    else:
        continue
fo.close()
