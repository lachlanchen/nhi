# -*- coding: UTF-8 -*-
from ctypes import *
import time
import argparse
import configparser
import os
from pprint import pprint
import csv



from datetime import datetime, timedelta


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

    def print_current_position(self):
        """
        Fetch and print the current XYZ positions.
        """
        machine_status = self.get_machine_status()
        if machine_status:
            print("Current XYZ Positions:")
            print(f"X: {machine_status.realPos[0]:.6f}")
            print(f"Y: {machine_status.realPos[1]:.6f}")
            print(f"Z: {machine_status.realPos[2]:.6f}")
        else:
            print("Failed to fetch current position.")

    def print_soft_origins(self):
        """
        Print the soft origins from the configuration file.
        """
        if 'ORIGIN' in self.config:
            print("Soft Origins:")
            print(f"X: {self.config['ORIGIN'].get('X', 'Not Set')}")
            print(f"Y: {self.config['ORIGIN'].get('Y', 'Not Set')}")
            print(f"Z: {self.config['ORIGIN'].get('Z', 'Not Set')}")
        else:
            print("Soft origins not set.")

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

    def move(self, axis, distance_mm, dir, speed, callback=None, record=True):
        # Adjust distance with scale factor and direction
        adjusted_distance = distance_mm * self.scale_factor * dir
        result = self.dll.FMC4030_Jog_Single_Axis(self.id, axis, c_float(adjusted_distance), c_float(speed), c_float(100), c_float(100), 1)
        if result != 0:
            print(f"Failed to move axis: Error {result}")
            return

        # # Wait for the movement to complete with optional callback for status updates
        # time.sleep(0.1)
        # while self.dll.FMC4030_Check_Axis_Is_Stop(self.id, axis) == 0:
        #     if callback:
        #         callback(self)
        #     time.sleep(0.1)

        # Use an internal method as callback to monitor position during movement
        if record:
            self._monitor_axis_position(axis)

    # def _monitor_axis_position(self, axis):
    #     """
    #     Monitor and print the current position of the specified axis during movement.
    #     """
    #     # Wait for the movement to complete with real-time status updates
    #     print(f"Moving axis {axis}...")
    #     while self.dll.FMC4030_Check_Axis_Is_Stop(self.id, axis) == 0:
    #         position = self.get_current_position(axis)
    #         print(f"Current position of axis {axis}: {position:.6f} mm")
    #         time.sleep(0.1)

    def _monitor_axis_position(self, axis):
        """
        Monitor and print the current position of the specified axis during movement and save to CSV.
        """
        # Specify the CSV file path
        csv_file_path = f"data/axis_{axis}_positions.csv"
        os.makedirs("data", exist_ok=True)

        # Check if the file already exists to decide whether to write the header
        file_exists = os.path.exists(csv_file_path)


        # Open the CSV file for writing
        with open(csv_file_path, mode='a', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)

            # Write the header to the CSV file
            # writer.writerow(['system_time', f'axis_{axis}_position_mm'])
            # Write the header to the CSV file if it doesn't exist
            if not file_exists:
                writer.writerow(['system_time', f'axis_{axis}_position_mm'])


            print(f"Moving axis {axis}...")
            while self.dll.FMC4030_Check_Axis_Is_Stop(self.id, axis) == 0:
                # Get the current position of the axis
                position = self.get_current_position(axis)

                # Capture the current system time formatted as hh:mm:ss,microseconds
                system_time_formatted = datetime.now().strftime("%H:%M:%S.%f")

                # Print the current position
                # print(f"Current position of axis {axis}: {position:.6f} mm")

                # Write the system time and position to the CSV file
                writer.writerow([system_time_formatted, f"{position:.6f}"])

                # Sleep for a short duration to limit the rate of updates
                time.sleep(0.001)

        print(f"Position data saved to {csv_file_path}")


    @property
    def is_axis_stop(self):


        return [self.dll.FMC4030_Check_Axis_Is_Stop(self.id, axis) for axis in range(MAX_AXIS)]
    

    def get_current_position(self, axis):
        position = c_float()
        result = self.dll.FMC4030_Get_Axis_Current_Pos(self.id, axis, byref(position))
        if result == 0:
            return position.value / self.scale_factor
        else:
            print(f"Failed to get current position: Error {result}")
            return None

    # def get_machine_status(self):
    #     result = self.dll.FMC4030_Get_Machine_Status(self.id, byref(self.machine_status))
    #     if result == 0:
    #         pprint(self.machine_status)

    #         self.machine_status.realPos = [pos/self.scale_factor for pos in self.machine_status.realPos]
    #         # self.machine_status = []
    #         return self.machine_status
    #     else:
    #         print(f"Failed to get machine status: Error {result}")
    #         return None

    def get_machine_status(self):
        result = self.dll.FMC4030_Get_Machine_Status(self.id, byref(self.machine_status))
        if result == 0:
            # Correctly adjust realPos values by dividing each by scale_factor
            for i in range(len(self.machine_status.realPos)):
                self.machine_status.realPos[i] = self.machine_status.realPos[i] / self.scale_factor

            # If you wish to print the modified realPos for verification or logging
            # print("Adjusted realPos values:")
            # for pos in self.machine_status.realPos:
            #     print(pos)

            return self.machine_status
        else:
            print(f"Failed to get machine status: Error {result}")
            return None

    def get_current_status(self):
        """
        Fetches the current machine status, focusing on x, y, z positions.
        Returns a dictionary with the axis names as keys and their positions as values.
        """
        status = self.get_machine_status()
        if status:
            # Assuming 'get_machine_status()' returns a MachineStatus instance
            # with 'realPos' attribute being a list [x_position, y_position, z_position].
            return self.is_axis_stop, {
                'x_position': status.realPos[0],
                'y_position': status.realPos[1],
                'z_position': status.realPos[2]
            }
        else:
            return self.is_axis_stop, {'x_position': 0, 'y_position': 0, 'z_position': 0}  # Default/fallback values



    # Define a callback function for real-time updates
    def get_callback(self, axis):

        def status_callback(self):
            position = self.get_current_position(axis)
            print(f"Current position of axis {axis}: {position} mm")
        return status_callback


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
                # self.move(i, abs(distance_to_move), -1 if distance_to_move < 0 else 1, 100, self.get_callback(i))
                self.move(i, abs(distance_to_move), -1 if distance_to_move < 0 else 1, 100)

    def detect_limits_and_set_origin(self):
        # Dictionary to hold the limit positions for each axis
        limit_positions = {'X': (None, None), 'Y': (None, None), 'Z': (None, None)}
        
        for axis in range(MAX_AXIS):
            # Move in the negative direction first
            self.move_until_limit(axis, -1)
            # Record the limit position
            neg_limit = self.get_current_position(axis)
            
            # Move in the positive direction
            self.move_until_limit(axis, 1)
            # Record the limit position
            pos_limit = self.get_current_position(axis)
            
            # Store the limits
            limit_positions[['X', 'Y', 'Z'][axis]] = (neg_limit, pos_limit)
            
            # Calculate and set the middle point as the soft origin
            middle_point = (neg_limit + pos_limit) / 2
            self.config['ORIGIN'][['X', 'Y', 'Z'][axis]] = str(middle_point)
        
        # Save the soft origins to the config file
        self.save_config()
        print("Limits detected and soft origin set:", limit_positions)

    def move_until_limit(self, axis, direction, distance=300, speed=100):
        """
        Move the specified axis in the given direction until the infrared sensor is triggered
        indicating the axis has hit its limit. Assumes a very large distance that is guaranteed
        to trigger the sensor.
        """
        # This is a placeholder for the actual movement code
        # The logic here needs to include checking the infrared sensor status
        # and stopping the movement when the limit is reached.
        self.move(axis, distance, direction, speed)
        # You'll need to implement logic to actually stop the movement based on the sensor.

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

# # Initialize the argument parser
# parser = argparse.ArgumentParser(description='Control the FMC4030 motor movement.')
# parser.add_argument('--axis', type=int, help='Axis number (0 for X, 1 for Y, 2 for Z)', required=True)
# parser.add_argument('--dir', type=int, choices=[-1, 1], help='Direction of movement (-1 for negative, 1 for positive)', required=True)
# parser.add_argument('--distance', type=float, help='Distance to move the motor in mm', required=True)
# parser.add_argument('--speed', type=float, help='Speed of the motor in mm/s', required=True)
# args = parser.parse_args()

parser = argparse.ArgumentParser(description='Control the FMC4030 motor movement.')
parser.add_argument('--axis', type=int, help='Axis number (0 for X, 1 for Y, 2 for Z)', required=False)
parser.add_argument('--dir', type=int, choices=[-1, 1], help='Direction of movement (-1 for negative, 1 for positive)', required=False)
parser.add_argument('--distance', type=float, help='Distance to move the motor in mm', required=False)
parser.add_argument('--speed', type=float, help='Speed of the motor in mm/s', default=20)
parser.add_argument('--set-origin', nargs='?', const=True, default=None, help="Set current position as origin. Provide 'xx,yy,zz' to move to new origin.")
parser.add_argument('--move', type=str, help='Move to specified position xx,yy,zz.', required=False)
args = parser.parse_args()

if __name__ == "__main__":
    dll_path = "FMC4030Lib-x64-20220329/FMC4030-Dll.dll"
    motor_system = MotorSystem(dll_path)

    # After any movements or configurations, print the current position and soft origins:
    motor_system.print_current_position()
    motor_system.print_soft_origins()

    # motor_system.detect_limits_and_set_origin()
    # motor_system.move_to_origin()

    # motor_system.print_current_position()
    # motor_system.print_soft_origins()



    if args.set_origin is not None:
        if args.set_origin is True:  # No coordinates provided, set current as origin
            motor_system.set_soft_origin()
        else:  # Coordinates provided, parse them
            coords = [float(c) for c in args.set_origin.split(',')]
            for i, coord in enumerate(coords):
                direction = 1 if coord >= 0 else -1
                motor_system.move(i, abs(coord), direction, args.speed)
            motor_system.set_soft_origin()

    if args.move:
        target_coords = [float(c) for c in args.move.split(',')]
        current_pos = motor_system.get_machine_status().realPos  # This should be adjusted as per your logic
        for i, coord in enumerate(target_coords):
            direction = 1 if coord - current_pos[i] >= 0 else -1
            distance = abs(coord - current_pos[i])
            motor_system.move(i, distance, direction, args.speed)

    if args.axis is not None and args.dir is not None and args.distance is not None:
        motor_system.move(args.axis, args.distance, args.dir, args.speed)

    motor_system.close_device()


# # Usage
# if __name__ == "__main__":
#     dll_path = "FMC4030Lib-x64-20220329/FMC4030-Dll.dll"
#     motor_system = MotorSystem(dll_path)
    

#     # # motor_system.fetch_device_parameters() # motor won't move when this line is here
#     # motor_system.fetch_device_parameters()
#     # time.sleep(1)  # Add a delay of 1 second

#     # print("Before fetching device parameters:")
#     # print(motor_system.get_machine_status())

#     motor_system.fetch_device_parameters()

#     # print("After fetching device parameters:")
#     # print(motor_system.get_machine_status())



#     # Set current position as origin
#     motor_system.set_soft_origin()

#     # motor_system.fetch_device_parameters() # motor will move when this line is here 
#     # motor_system.fetch_device_parameters() # motor won't move when this line is here
#     # motor_system.fetch_device_parameters()
#     # time.sleep(1)  # Add a delay of 1 second

#     # print("Before fetching device parameters:")
#     # print(motor_system.get_machine_status())

#     # motor_system.fetch_device_parameters()

#     # print("After fetching device parameters:")
#     # print(motor_system.get_machine_status())

#     # Perform random movements
#     motor_system.move(0, 50, -1, args.speed)  # Move X axis
#     # motor_system.move(1, 50, -1, args.speed)  # Move Y axis
#     # motor_system.move(2, 15, -1, args.speed)  # Move Z axis

#     # Return to origin
#     motor_system.move_to_origin()

#     motor_system.fetch_device_parameters() # motor will move then whis line is here
#     # motor_system.fetch_device_parameters() # motor won't move when this line is here
#     # motor_system.fetch_device_parameters()
#     # time.sleep(1)  # Add a delay of 1 second

#     # print("Before fetching device parameters:")
#     # print(motor_system.get_machine_status())

#     # motor_system.fetch_device_parameters()

#     # print("After fetching device parameters:")
#     # print(motor_system.get_machine_status())

#     motor_system.close_device()
