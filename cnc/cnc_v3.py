# -*- coding: UTF-8 -*-
from ctypes import *
import time
import argparse

# Assuming MAX_AXIS is 3 as per the header file
MAX_AXIS = 3

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

class MachineStatus(Structure):
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

class MotorSystem:
    def __init__(self, dll_path, ip="192.168.0.30", port=8088, scale_factor=10):
        self.dll = windll.LoadLibrary(dll_path)
        self.ip = ip.encode('utf-8')
        self.port = port
        self.id = 0  # Assuming a single device for simplicity
        self.device_params = DeviceParams()
        self.machine_status = MachineStatus()
        self.scale_factor = scale_factor  # Scale factor to adjust actual movement distance
        self.open_device()

    def open_device(self):
        result = self.dll.FMC4030_Open_Device(self.id, c_char_p(self.ip), self.port)
        if result != 0:
            print(f"Failed to open device: Error {result}")

    def close_device(self):
        result = self.dll.FMC4030_Close_Device(self.id)
        if result != 0:
            print(f"Failed to close device: Error {result}")

    def move(self, axis, distance_mm, dir, speed, callback=None):
        # Adjust distance with scale factor and direction
        adjusted_distance = distance_mm * self.scale_factor * dir
        result = self.dll.FMC4030_Jog_Single_Axis(self.id, axis, c_float(adjusted_distance), c_float(speed), c_float(100), c_float(100), 1)
        if result != 0:
            print(f"Failed to move axis: Error {result}")
            return

        # Polling for the movement to complete in a non-blocking manner
        def check_movement():
            while self.dll.FMC4030_Check_Axis_Is_Stop(self.id, axis) == 0:
                if callback:
                    callback(self)
                time.sleep(0.1)

        # Use a separate thread to avoid blocking
        from threading import Thread
        movement_thread = Thread(target=check_movement)
        movement_thread.start()

    def get_current_position(self, axis):
        position = c_float()
        result = self.dll.FMC4030_Get_Axis_Current_Pos(self.id, axis, byref(position))
        if result == 0:
            return position.value
        else:
            print(f"Failed to get current position: Error {result}")
            return None

    def get_machine_status(self):
        result = self.dll.FMC4030_Get_Machine_Status(self.id, byref(self.machine_status))
        if result == 0:
            return self.machine_status
        else:
            print(f"Failed to get machine status: Error {result}")
            return None

    def fetch_device_parameters(self):
        result = self.dll.FMC4030_Get_Device_Para(self.id, byref(self.device_params))
        if result == 0:
            print("Device Parameters:")
            print(f"ID: {self.device_params.id}")
            # Additional prints for new parameters as in cnc_v2.py
        else:
            print("Failed to get device parameters")

# Define a callback function for real-time updates
def status_callback(motor_system):
    position = motor_system.get_current_position(args.axis)
    print(f"Current position of axis {args.axis}: {position} mm")

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Control the FMC4030 motor movement.')
parser.add_argument('--axis', type=int, help='Axis number (0 for X, 1 for Y, 2 for Z)', required=True)
parser.add_argument('--dir', type=int, choices=[-1, 1], help='Direction of movement (-1 for negative, 1 for positive)', required=True)
parser.add_argument('--distance', type=float, help='Distance to move the motor in mm', required=True)
parser.add_argument('--speed', type=float, help='Speed of the motor in mm/s', required=True)
args = parser.parse_args()

# Usage
if __name__ == "__main__":
    dll_path = "FMC4030Lib-x64-20220329/FMC4030-Dll.dll"
    motor_system = MotorSystem(dll_path)
    motor_system.fetch_device_parameters()
    motor_system.move(args.axis, args.distance, args.dir, args.speed, status_callback)
    motor_system.close_device()
