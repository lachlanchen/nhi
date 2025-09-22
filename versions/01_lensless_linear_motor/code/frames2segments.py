import pandas as pd
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import cv2  # OpenCV for video handling

class CNCPositionInterpolation:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = self.load_csv()
        self.interpolator = self.create_interpolator()

    def load_csv(self):
        df = pd.read_csv(self.csv_path)
        df['system_time'] = pd.to_datetime(df['system_time'], format='%H:%M:%S.%f')
        return df

    def create_interpolator(self):
        x = self.data['system_time'].astype(int)  # Convert times to integers for interpolation
        y = self.data['axis_1_position_mm']
        return interp1d(x, y, bounds_error=False, fill_value=(y.iloc[0], y.iloc[-1]))

    def get_position(self, timestamp_str):
        timestamp = datetime.strptime(timestamp_str, '%H:%M:%S.%f')
        position = self.interpolator(pd.Timestamp(timestamp).value)
        return position

class FramePositionIntegrator:
    def __init__(self, frames_csv, position_interpolator):
        self.frames_csv = frames_csv
        self.position_interpolator = position_interpolator
        self.frames = self.load_frame_timestamps()

    def load_frame_timestamps(self):
        df = pd.read_csv(self.frames_csv)
        df['frame_timestamp'] = pd.to_datetime(df['frame_timestamp'], format='%H:%M:%S.%f')
        return df

    def add_positions_to_frames(self):
        self.frames['position'] = self.frames['frame_timestamp'].apply(
            lambda x: self.position_interpolator.get_position(x.strftime('%H:%M:%S.%f')))
        return self.frames

class VideoSegmenter:
    def __init__(self, video_data, frame_positions):
        self.video_data = np.load(video_data)  # Assuming shape (N, H, W, C)
        self.frame_positions = frame_positions

    def segment_and_save(self, output_folder):
        current_segment = []
        start_pos = self.frame_positions.iloc[0]['position']
        for index, row in self.frame_positions.iterrows():
            frame = self.video_data[index]
            if row['position'] < start_pos or index == len(self.frame_positions) - 1:  # New segment condition
                self.save_segment(current_segment, output_folder, int(start_pos), int(row['position']))
                current_segment = []
                start_pos = row['position']
            current_segment.append(frame)

    def save_segment(self, frames, folder, start_index, end_index):
        out_path = f"{folder}/segment_{start_index}_to_{end_index}.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            out.write(frame.astype('uint8'))
        out.release()

# Example Usage:
cnc = CNCPositionInterpolation('axis_1_positions.csv')
frames = FramePositionIntegrator('frames_output_timestamps.csv', cnc)
frame_positions = frames.add_positions_to_frames()

video_segmenter = VideoSegmenter('frames_output.npy', frame_positions)
video_segmenter.segment_and_save('output_videos')

