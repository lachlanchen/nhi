import pandas as pd
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
from ctypes import c_float
import matplotlib.pyplot as plt

class CNCMotorSystem:
    def __init__(self, csv_path, scale_factor=10):
        self.csv_path = csv_path
        # self.id = id
        # self.dll = dll
        self.scale_factor = scale_factor
        self.data = self.load_csv()
        self.fitter = self.fit_data()
        
    def load_csv(self):
        df = pd.read_csv(self.csv_path)
        df['system_time'] = pd.to_datetime(df['system_time'], format='%H:%M:%S.%f')
        df['time_microseconds'] = df['system_time'].apply(lambda x: x.microsecond + x.second * 1000000 + x.minute * 60000000 + x.hour * 3600000000)
        return df
    
    def fit_data(self):
        x = self.data['time_microseconds']
        y = self.data['axis_1_position_mm']
        return interp1d(x, y, fill_value="extrapolate")
    
    def predict_position(self, timestamp, offset=0):
        datetime_obj = datetime.strptime(timestamp, '%H:%M:%S.%f')
        microseconds = datetime_obj.microsecond + datetime_obj.second * 1000000 + datetime_obj.minute * 60000000 + datetime_obj.hour * 3600000000
        microseconds += offset * 1000000
        return self.fitter(microseconds)
    
    # def move(self, axis, distance_mm, dir, speed, callback=None):
    #     adjusted_distance = distance_mm * self.scale_factor * dir
    #     result = self.dll.FMC4030_Jog_Single_Axis(self.id, axis, c_float(adjusted_distance), c_float(speed), c_float(100), c_float(100), 1)
    #     if result != 0:
    #         print(f"Failed to move axis: Error {result}")
    #         return
    #     # Assuming _monitor_axis_position is implemented elsewhere
    #     self._monitor_axis_position(axis)

    def plot_and_save_fit_curve(self, filename):
        x = self.data['time_microseconds']
        y = self.data['axis_1_position_mm']
        
        # Generate points for the fit curve
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = self.fitter(x_fit)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'bo', label='Original Data')
        plt.plot(x_fit, y_fit, 'r-', label='Fitted Curve')
        plt.xlabel('Time (Microseconds)')
        plt.ylabel('Axis 1 Position (mm)')
        plt.title('CNC Motor System Y Axis Position vs. Time')
        plt.legend()
        plt.grid(True)

        # Save plot
        plt.savefig(filename)
        plt.close()
if __name__ == '__main__':
    # Example usage (assuming `dll` is your library object):
    motor_system = CNCMotorSystem('data/axis_1_positions.csv')
    print(motor_system.predict_position('20:14:45.000000'))
    # Assuming the CNCMotorSystem instance is already created as `motor_system`
    motor_system.plot_and_save_fit_curve('fit_curve_plot.png')