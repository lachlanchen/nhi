# -*- coding: UTF-8 -*-
from ctypes import *
import time
import argparse

# Import the library for handling command line arguments
import argparse

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Control the FMC4030 motor movement.')
parser.add_argument('--axis', type=int, help='Axis number (0 for X, 1 for Y, 2 for Z)', required=True)
parser.add_argument('--dir', type=int, choices=[-1, 1], help='Direction of movement (-1 for negative, 1 for positive)', required=True)
parser.add_argument('--speed', type=float, help='Speed of the motor in mm/s', required=True)
parser.add_argument('--distance', type=float, help='Distance to move the motor in mm', required=True)

# Parse the arguments
args = parser.parse_args()

fmc4030 = windll.LoadLibrary('C:/Users/lachlan/Desktop/FMC4030-2022-06-19/English/FMC4030Lib-x64-20220322/FMC4030-Dll.dll')

class machine_status(Structure):
    _fields_ = [
        ("realPos", c_float * 3),
        ("realSpeed", c_float * 3),
        ("inputStatus", c_int32),
        ("outputStatus", c_int32),
        ("limitNStatus", c_int32),
        ("limitPStatus", c_int32),
        ("machineRunStatus", c_int32),
        ("axisStatus", c_int32 * 3),
        ("homeStatus", c_int32),
        ("file", c_ubyte * 600)
    ]

ms = machine_status()

id = 0
axis = args.axis  # Axis is now set by command line argument
ip = "192.168.0.30"
port = 8088

print(fmc4030.FMC4030_Open_Device(id, c_char_p(bytes(ip, 'utf-8')), port))

# The distance to move is determined by the direction and distance argument
move_distance = args.distance * args.dir

print(fmc4030.FMC4030_Jog_Single_Axis(id, axis, c_float(move_distance), c_float(args.speed), c_float(100), c_float(100), 1))

time.sleep(0.1)

while fmc4030.FMC4030_Check_Axis_Is_Stop(id, axis) == 0:
    fmc4030.FMC4030_Get_Machine_Status(id, pointer(ms))
    print(ms.realPos[axis])

print(fmc4030.FMC4030_Close_Device(id))
