# -*- coding: UTF-8 -*-
from ctypes import *
import time
import socket

#fmc4030 = windll.LoadLibrary('G:/Project/AMC4030_V2/DLL/FMC4030-Dll/Release/FMC4030-Dll.dll')
fmc4030 = windll.LoadLibrary('C:/Users/lachlan/Desktop/FMC4030-2022-06-19/English/FMC4030Lib-x64-20220322/FMC4030-Dll.dll')

#定义设备状态类，用于获取设备状态数据
#struct machine_status{
# 	float realPos[3];
# 	float realSpeed[3];
# 	unsigned int inputStatus;
# 	unsigned int outputStatus;
# 	unsigned int limitNStatus;
# 	unsigned int limitPStatus;
# 	unsigned int machineRunStatus;
# 	unsigned int axisStatus[MAX_AXIS];
# 	unsigned int homeStatus;
# 	char file[20][30];
# };
class machine_status(Structure):
    _fields_ = [
        ("realPos", c_float * 3),
        ("realSpeed", c_float * 3),
        ("inputStatus", c_int32 * 1),
        ("outputStatus", c_int32 * 1),
        ("limitNStatus", c_int32 * 1),
        ("limitPStatus", c_int32 * 1),
        ("machineRunStatus", c_int32 * 1),
        ("axisStatus", c_int32 * 3),
        ("homeStatus", c_int32 * 1),
        ("file", c_ubyte * 600)
    ]

ms = machine_status()

#给控制器编号，此ID号唯一
id = 0
axis = 1
ip = "192.168.0.30"
port = 8088

#连接控制器
print (fmc4030.FMC4030_Open_Device(id, c_char_p(bytes("192.168.0.30", 'utf-8')), port))

#控制器单轴运动
print (fmc4030.FMC4030_Jog_Single_Axis(id, axis, c_float(1000), c_float(100), c_float(100), c_float(100), 1))

#延时等待，等待控制卡实际启动
time.sleep(0.1)

#等待轴运行完成，过程中不断获取轴实际位置并输出
while fmc4030.FMC4030_Check_Axis_Is_Stop(id, axis) == 0:
    fmc4030.FMC4030_Get_Machine_Status(id, pointer(ms))
    print (ms.realPos[axis])

#关闭控制器连接，使用完成一定调用此函数释放资源
print (fmc4030.FMC4030_Close_Device(id))
