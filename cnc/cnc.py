# -*- coding: UTF-8 -*-
from ctypes import *
import time
import argparse

MAX_AXIS = 3

# Define the DeviceParams structure
class DeviceParams(Structure):
    _fields_ = [
        ("id", c_uint),
        ("bound232", c_uint),
        ("bound485", c_uint),
        ("ip", c_char * 15),
        ("port", c_int),
        ("div", c_int * MAX_AXIS),
        ("lead", c_int * MAX_AXIS),
        ("softLimitMax", c_int * MAX_AXIS),
        ("softLimitMin", c_int * MAX_AXIS),
        ("homeTime", c_int * MAX_AXIS),
    ]

# Define the machine_status structure
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

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Control the FMC4030 motor movement.')
parser.add_argument('--axis', type=int, help='Axis number (0 for X, 1 for Y, 2 for Z)', required=True)
parser.add_argument('--dir', type=int, choices=[-1, 1], help='Direction of movement (-1 for negative, 1 for positive)', required=True)
parser.add_argument('--speed', type=float, help='Speed of the motor in mm/s', required=True)
parser.add_argument('--distance', type=float, help='Distance to move the motor in mm', required=True)

# Parse the arguments
args = parser.parse_args()

# Load the DLL and declare function argument and return types
fmc4030 = windll.LoadLibrary('C:/Users/lachlan/Desktop/FMC4030-2022-06-19/English/FMC4030Lib-x64-20220322/FMC4030-Dll.dll')
fmc4030.FMC4030_Get_Device_Para.argtypes = [c_int, POINTER(DeviceParams)]
fmc4030.FMC4030_Get_Device_Para.restype = c_int

# Instantiate structures
ms = machine_status()
device_params = DeviceParams()

# Parameters for device
id = 0
axis = args.axis
ip = "192.168.0.30"
port = 8088

# Open device
print(fmc4030.FMC4030_Open_Device(id, c_char_p(bytes(ip, 'utf-8')), port))



# Calculate and move the motor
move_distance = args.distance * args.dir
print(fmc4030.FMC4030_Jog_Single_Axis(id, axis, c_float(move_distance), c_float(args.speed), c_float(100), c_float(100), 1))

time.sleep(0.1)

while fmc4030.FMC4030_Check_Axis_Is_Stop(id, axis) == 0:
    fmc4030.FMC4030_Get_Machine_Status(id, pointer(ms))
    print(ms.realPos[axis])

# Fetch the device parameters
result = fmc4030.FMC4030_Get_Device_Para(id, byref(device_params))

if result == 0:  # Assuming 0 is the success code
    print(f"Lead: {device_params.lead}")
    print(f"Microstepping: {device_params.div}")
else:
    print("Failed to get device parameters")

# Close device
print(fmc4030.FMC4030_Close_Device(id))
