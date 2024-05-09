import pandas as pd
import os
from timestamp2position import CNCMotorSystem
import argparse

class DataWithPosition:
    def __init__(self, folder_path, csv_name, timestamp_column_name):
        self.timestamp_column_name = timestamp_column_name
        # Constructor initializes the main attributes for the class
        self.folder_path = folder_path  # Path to the data folder
        self.csv_path = os.path.join(folder_path, csv_name)  # Full path to the CSV file to be processed
        self.axis_csv_path = os.path.join(folder_path, 'axis_1_positions.csv')  # Path to the CNC axis positions file
        self.motor_system = CNCMotorSystem(self.axis_csv_path)  # Initializes the CNCMotorSystem with the axis CSV path
        # Generate a plot of the fitted curve using the motor system data and save it
        self.motor_system.plot_and_save_fit_curve(os.path.join(folder_path, 'fit_curve_plot.png'))
        self.filtered_data = None  # Placeholder for storing filtered data
        self.offset = 0  # Initial offset is set to zero microseconds
        self.set_offset()  # Adjust the offset before filtering

    def set_offset(self):
        # Method to set the time offset based on the first entry in the provided CSV file
        data = pd.read_csv(self.csv_path)
        if not data.empty:
            # Calculate offset from the first row's timestamps
            first_data_time = pd.Timestamp(data.iloc[0][self.timestamp_column_name])
            first_system_time = pd.Timestamp(data.iloc[0]['system_timestamp'])
            # Offset in microseconds
            self.offset = (first_system_time - first_data_time).total_seconds()

    def timestamp_to_microseconds(self, timestamp):
        # Converts a timestamp in HH:MM:SS.ffffff format into microseconds since the start of the day, adjusted by offset
        parts = timestamp.split(':')
        hours, minutes = int(parts[0]), int(parts[1])
        seconds, microseconds = map(int, parts[2].split('.'))
        return (hours * 3600 + minutes * 60 + seconds) * 1000000 + microseconds

    def is_within_range(self, data_time, start_time, end_time):
        # Check if a given timestamp falls within the specified start and end times
        data_time_microseconds = self.timestamp_to_microseconds(data_time) + self.offset * 1e6
        start_time_microseconds = self.timestamp_to_microseconds(start_time)
        end_time_microseconds = self.timestamp_to_microseconds(end_time)
        return start_time_microseconds <= data_time_microseconds <= end_time_microseconds

    def load_and_filter_data(self, ):
        # Loads data and filters entries based on whether their timestamps fall within the range of the CNC system data
        timestamp_column_name = self.timestamp_column_name

        axis_data = pd.read_csv(self.axis_csv_path)
        start_time = axis_data['system_time'].min()
        end_time = axis_data['system_time'].max()
        data = pd.read_csv(self.csv_path)
        self.filtered_data = data[data[timestamp_column_name].apply(
            lambda x: self.is_within_range(x, start_time, end_time))]

    def predict_positions(self, ):
        timestamp_column_name = self.timestamp_column_name

        # Predicts and appends the CNC axis position to each data entry based on its timestamp
        if self.filtered_data is not None:
            self.filtered_data['axis_y_position_mm'] = self.filtered_data[timestamp_column_name].apply(
                lambda x: self.motor_system.predict_position(x, offset=self.offset))
        else:
            print("Data is not loaded. Please run load_and_filter_data() first.")

    def save_filtered_data(self, output_filename):
        # Saves the filtered data with predicted positions to a specified file in the data folder
        if self.filtered_data is not None:
            final_output_path = os.path.join(self.folder_path, output_filename)
            self.filtered_data.to_csv(final_output_path, index=False)
            print(f"Filtered data saved successfully to {final_output_path}.")
        else:
            print("Data is not loaded. Please run load_and_filter_data() and predict_positions() first.")

class EventDataWithPosition(DataWithPosition):
    def __init__(self, folder_path, timestamp_column_name):
        # Initialize the class specifically for handling event data
        super().__init__(folder_path, 'events_output.csv', timestamp_column_name)

class FrameDataWithPosition(DataWithPosition):
    def __init__(self, folder_path, timestamp_column_name):
        # Initialize the class specifically for handling frame data
        super().__init__(folder_path, 'frames_output_timestamps.csv', timestamp_column_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process event or frame data with position interpolation.")
    parser.add_argument('-t', '--type', choices=['event', 'frame'], default='event',
                        help='Type of data to process: event or frame.')
    parser.add_argument('-p', '--path', default='data',
                        help='Path to the data folder.')
    
    args = parser.parse_args()

    timestamp_column_name = 'event_timestamp' if args.type == 'event' else 'frame_timestamp'
    output_filename = 'filtered_{}_data_with_positions.csv'.format(args.type)
    if args.type == 'event':
        processor = EventDataWithPosition(args.path, timestamp_column_name)
    else:
        processor = FrameDataWithPosition(args.path, timestamp_column_name)

    processor.load_and_filter_data()
    processor.predict_positions()
    processor.save_filtered_data(output_filename)
