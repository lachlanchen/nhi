import pandas as pd
from timestamp2position import CNCMotorSystem
import os

class FrameDataWithPosition:
    def __init__(self, folder_path, offset=-80):
        self.folder_path = folder_path
        # Automatically assign the expected filenames within the given folder
        self.axis_csv_path = os.path.join(folder_path, 'axis_1_positions.csv')
        self.frame_csv_path = os.path.join(folder_path, 'frames_output_timestamps.csv')
        # Initialize the motor system with the path to axis CSV
        self.motor_system = CNCMotorSystem(self.axis_csv_path)
        # Generate and save the fitting curve plot
        self.motor_system.plot_and_save_fit_curve('fit_curve_plot.png')
        self.filtered_frames = None

        self.offset = offset

        # self.folder_path = os.path.join(self.folder_path, "intermediate")
        # os.makedirs(self.folder_path, exist_ok=True)

    def timestamp_to_microseconds(self, timestamp):
        # Convert a timestamp string (HH:MM:SS.ffffff) into microseconds
        parts = timestamp.split(':')
        hours, minutes = int(parts[0]), int(parts[1])
        seconds, microseconds = map(int, parts[2].split('.'))
        return (hours * 3600 + minutes * 60 + seconds) * 1000000 + microseconds

    def is_within_range(self, frame_time, start_time, end_time):
        # Convert string times to numeric microseconds and compare
        frame_time_microseconds = self.timestamp_to_microseconds(frame_time) + self.offset * 1000000
        start_time_microseconds = self.timestamp_to_microseconds(start_time)
        end_time_microseconds = self.timestamp_to_microseconds(end_time)
        return start_time_microseconds <= frame_time_microseconds <= end_time_microseconds
        # return True

    def load_and_filter_frames(self):
        # Load axis data and determine the time range for filtering frames
        axis_data = pd.read_csv(self.axis_csv_path)
        start_time = axis_data['system_time'].min()
        end_time = axis_data['system_time'].max()

        # Load frames and filter based on the time range determined from axis data
        frame_data = pd.read_csv(self.frame_csv_path)
        self.filtered_frames = frame_data[frame_data['frame_timestamp'].apply(
            lambda x: self.is_within_range(x, start_time, end_time))]

    def predict_positions(self):
        # Predict the position for each frame timestamp if frames are loaded and filtered
        if self.filtered_frames is not None:
            self.filtered_frames['axis_y_position_mm'] = self.filtered_frames['frame_timestamp'].apply(
                lambda x: self.motor_system.predict_position(x, offset=self.offset)
            )
        else:
            print("Filtered frames are not loaded. Please run load_and_filter_frames() first.")

    def save_filtered_data(self, output_filename='filtered_frame_data_with_positions.csv'):
        # Save the filtered and augmented data to a CSV file in the specified folder
        if self.filtered_frames is not None:
            final_output_path = os.path.join(self.folder_path, output_filename)
            final_data = self.filtered_frames.drop(columns=['system_timestamp'])
            final_data.to_csv(final_output_path, index=False)
            print(f"Filtered and augmented frame data saved successfully to {final_output_path}.")
        else:
            print("Filtered frames are not loaded. Please run load_and_filter_frames() and predict_positions() first.")

if __name__ == '__main__':
    # Initialize the class with the folder path containing the CSV files
    frame_data_processor = FrameDataWithPosition('data')
    # Load and filter frames according to axis time range
    frame_data_processor.load_and_filter_frames()
    # Predict axis positions based on frame timestamps
    frame_data_processor.predict_positions()
    # Optionally specify an output filename, or use the default
    frame_data_processor.save_filtered_data()
