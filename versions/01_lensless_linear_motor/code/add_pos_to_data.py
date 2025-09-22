import pandas as pd
import os
from timestamp2position import CNCMotorSystem
import argparse

class DataWithPosition:
    def __init__(self, folder_path, csv_name, timestamp_column_name, offset=None):
        self.timestamp_column_name = timestamp_column_name
        self.folder_path = folder_path
        self.csv_path = os.path.join(folder_path, csv_name)
        self.axis_csv_path = os.path.join(folder_path, 'axis_1_positions.csv')
        self.motor_system = CNCMotorSystem(self.axis_csv_path)
        self.motor_system.plot_and_save_fit_curve(os.path.join(folder_path, 'fit_curve_plot.png'))
        self.filtered_data = None
        self.offset = offset  # Set directly if provided, otherwise calculate
        if self.offset is None:
            self.set_offset()
        else:
            print(f"Using provided offset: {self.offset} seconds.")

    def set_offset(self):
        data = pd.read_csv(self.csv_path)
        if not data.empty:
            first_data_time = pd.Timestamp(data.iloc[0][self.timestamp_column_name])
            if 'system_timestamp' in data.columns:
                first_system_time = pd.Timestamp(data.iloc[0]['system_timestamp'])
            else:
                first_system_time = first_data_time
            self.offset = (first_system_time - first_data_time).total_seconds()
            print(f"Calculated offset: {self.offset} seconds.")

    def timestamp_to_microseconds(self, timestamp):
        parts = timestamp.split(':')
        hours, minutes = int(parts[0]), int(parts[1])
        seconds, microseconds = map(int, parts[2].split('.'))
        return (hours * 3600 + minutes * 60 + seconds) * 1000000 + microseconds

    def is_within_range(self, data_time, start_time, end_time):
        data_time_microseconds = self.timestamp_to_microseconds(data_time) + self.offset * 1e6
        start_time_microseconds = self.timestamp_to_microseconds(start_time)
        end_time_microseconds = self.timestamp_to_microseconds(end_time)
        return start_time_microseconds <= data_time_microseconds <= end_time_microseconds

    def load_and_filter_data(self):
        axis_data = pd.read_csv(self.axis_csv_path)
        start_time = axis_data['system_time'].min()
        end_time = axis_data['system_time'].max()
        data = pd.read_csv(self.csv_path)
        self.filtered_data = data[data[self.timestamp_column_name].apply(
            lambda x: self.is_within_range(x, start_time, end_time))]

    def predict_positions(self):
        if self.filtered_data is not None:
            self.filtered_data['axis_y_position_mm'] = self.filtered_data[self.timestamp_column_name].apply(
                lambda x: self.motor_system.predict_position(x, offset=self.offset))
        else:
            print("Data is not loaded. Please run load_and_filter_data() first.")

    def save_filtered_data(self, output_filename):
        if self.filtered_data is not None:
            final_output_path = os.path.join(self.folder_path, output_filename)
            self.filtered_data.to_csv(final_output_path, index=False)
            print(f"Filtered data saved successfully to {final_output_path}.")
        else:
            print("Data is not loaded. Please run load_and_filter_data() and predict_positions() first.")

class EventDataWithPosition(DataWithPosition):
    def __init__(self, folder_path, timestamp_column_name, offset=None):
        super().__init__(folder_path, 'events_output.csv', timestamp_column_name, offset)

class FrameDataWithPosition(DataWithPosition):
    def __init__(self, folder_path, timestamp_column_name, offset=None):
        super().__init__(folder_path, 'frames_output_timestamps.csv', timestamp_column_name, offset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process event or frame data with position interpolation.")
    parser.add_argument('-t', '--type', choices=['event', 'frame'], default='event',
                        help='Type of data to process: event or frame.')
    parser.add_argument('-p', '--path', default='data',
                        help='Path to the data folder.')
    parser.add_argument('-o', '--offset', type=float, help='Optional offset in seconds to bypass calculation.')

    args = parser.parse_args()

    timestamp_column_name = 'event_timestamp' if args.type == 'event' else 'frame_timestamp'
    output_filename = f'filtered_{args.type}_data_with_positions.csv'
    if args.type == 'event':
        processor = EventDataWithPosition(args.path, timestamp_column_name, args.offset)
    else:
        processor = FrameDataWithPosition(args.path, timestamp_column_name, args.offset)

    processor.load_and_filter_data()
    processor.predict_positions()
    processor.save_filtered_data(output_filename)
