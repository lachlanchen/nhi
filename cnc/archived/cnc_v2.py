# -*- coding: UTF-8 -*-
from ctypes import *
import time
import argparse
import configparser
import os

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
    def __init__(self, dll_path, ip="192.168.0.30", port=8088, scale_factor=10, config_file='motor_system.ini'):
        self.dll = windll.LoadLibrary(dll_path)
        self.ip = ip.encode('utf-8')
        self.port = port
        self.id = 0  # Assuming a single device for simplicity
        self.device_params = DeviceParams()
        self.machine_status = MachineStatus()
        self.scale_factor = scale_factor  # Scale factor to adjust actual movement distance
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_config()
        self.open_device()

    def load_config(self):
        if not os.path.exists(self.config_file):
            self.config['ORIGIN'] = {'X': '0', 'Y': '0', 'Z': '0'}
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
        self.config.read(self.config_file)

    def save_config(self):
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

    def open_device(self):
        result = self.dll.FMC4030_Open_Device(self.id, c_char_p(self.ip), self.port)
        if result != 0:
            print(f"Failed to open device: Error {result}")

    def close_device(self):
        result = self.dll.FMC4030_Close_Device(self.id)
        if result != 0:
            print(f"Failed to close device: Error {result}")

    def restart_device(self):
        self.close_device()
        self.open_device()

    def move(self, axis, distance_mm, dir, speed, callback=None):
        # self.restart_device()

        # Adjust distance with scale factor and direction
        adjusted_distance = distance_mm * self.scale_factor * dir
        result = self.dll.FMC4030_Jog_Single_Axis(self.id, axis, c_float(adjusted_distance), c_float(speed), c_float(100), c_float(100), 1)
        if result != 0:
            print(f"Failed to move axis: Error {result}")
            return

        # Wait for the movement to complete with optional callback for status updates
        time.sleep(0.1)
        while self.dll.FMC4030_Check_Axis_Is_Stop(self.id, axis) == 0:
            if callback:
                callback(self)
            time.sleep(0.1)

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

    def set_soft_origin(self):
        # Fetch the current machine status
        machine_status = self.get_machine_status()
        if machine_status is not None:
            # Store the current positions as soft origin
            for i, axis in enumerate(['X', 'Y', 'Z']):
                self.config['ORIGIN'][axis] = str(machine_status.realPos[i])
            self.save_config()

    def move_to_origin(self):
        # Fetch the soft origin from config
        origin = {axis: float(self.config['ORIGIN'][axis]) for axis in ['X', 'Y', 'Z']}
        # Fetch the current machine status
        machine_status = self.get_machine_status()
        if machine_status is not None:
            for i, axis in enumerate(['X', 'Y', 'Z']):
                # Calculate the distance to move back to the origin
                distance_to_move = origin[axis] - machine_status.realPos[i]
                # Move each axis back to the origin
                self.move(i, abs(distance_to_move)/self.scale_factor, -1 if distance_to_move < 0 else 1, 100)

    # def fetch_device_parameters(self):
    #     result = self.dll.FMC4030_Get_Device_Para(self.id, byref(self.device_params))
    #     if result == 0:
    #         print(f"Lead: {self.device_params.lead}")
    #         print(f"Microstepping: {self.device_params.div}")
    #     else:
    #         print("Failed to get device parameters")

    # def fetch_device_parameters(self):
    #     result = self.dll.FMC4030_Get_Device_Para(self.id, byref(self.device_params))
    #     if result == 0:
    #         # Using list comprehension to print array contents
    #         lead_values = [self.device_params.lead[i] for i in range(MAX_AXIS)]
    #         print(f"Lead: {lead_values}")

    #         div_values = [self.device_params.div[i] for i in range(MAX_AXIS)]
    #         print(f"Microstepping (div): {div_values}")

    #         # Alternatively, for a more detailed output:
    #         print("Lead values per axis:")
    #         for i in range(MAX_AXIS):
    #             print(f"Axis {i+1}: Lead = {self.device_params.lead[i]}, Microstepping (div) = {self.device_params.div[i]}")
    #     else:
    #         print("Failed to get device parameters")

    def fetch_device_parameters(self):
        result = self.dll.FMC4030_Get_Device_Para(self.id, byref(self.device_params))
        if result == 0:
            print("Device Parameters:")
            print(f"ID: {self.device_params.id}")
            print(f"Bound232: {self.device_params.bound232}")
            print(f"Bound485: {self.device_params.bound485}")
            print(f"IP: {self.device_params.ip.decode('utf-8')}")
            print(f"Port: {self.device_params.port}")
            for axis in range(MAX_AXIS):
                print(f"Axis {axis} Div: {self.device_params.div[axis]}")
                print(f"Axis {axis} Lead: {self.device_params.lead[axis]}")
                print(f"Axis {axis} Soft Limit Max: {self.device_params.softLimitMax[axis]}")
                print(f"Axis {axis} Soft Limit Min: {self.device_params.softLimitMin[axis]}")
                print(f"Axis {axis} Home Time: {self.device_params.homeTime[axis]}")
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
    

    # # motor_system.fetch_device_parameters() # motor won't move when this line is here
    # motor_system.fetch_device_parameters()
    # time.sleep(1)  # Add a delay of 1 second

    # print("Before fetching device parameters:")
    # print(motor_system.get_machine_status())

    motor_system.fetch_device_parameters()

    # print("After fetching device parameters:")
    # print(motor_system.get_machine_status())



    # Set current position as origin
    motor_system.set_soft_origin()

    # motor_system.fetch_device_parameters() # motor will move when this line is here 
    # motor_system.fetch_device_parameters() # motor won't move when this line is here
    # motor_system.fetch_device_parameters()
    # time.sleep(1)  # Add a delay of 1 second

    # print("Before fetching device parameters:")
    # print(motor_system.get_machine_status())

    # motor_system.fetch_device_parameters()

    # print("After fetching device parameters:")
    # print(motor_system.get_machine_status())

    # Perform random movements
    motor_system.move(0, 50, 1, args.speed, status_callback)  # Move X axis
    # motor_system.move(1, 50, -1, args.speed, status_callback)  # Move Y axis
    # motor_system.move(2, 15, -1, args.speed, status_callback)  # Move Z axis

    # Return to origin
    # motor_system.move_to_origin()

    motor_system.fetch_device_parameters() # motor will move then whis line is here
    # motor_system.fetch_device_parameters() # motor won't move when this line is here
    # motor_system.fetch_device_parameters()
    # time.sleep(1)  # Add a delay of 1 second

    # print("Before fetching device parameters:")
    # print(motor_system.get_machine_status())

    # motor_system.fetch_device_parameters()

    # print("After fetching device parameters:")
    # print(motor_system.get_machine_status())

    motor_system.close_device()
