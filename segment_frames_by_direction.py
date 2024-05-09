import numpy as np
import pandas as pd
import cv2
import os

class NpyToVideoConverter:
    def __init__(self, folder_path, output_folder='video_output', fps=1):
        self.folder_path = folder_path
        self.output_folder = os.path.join(folder_path, output_folder)
        os.makedirs(self.output_folder, exist_ok=True)
        self.fps = fps

    def convert_npy_to_video(self, npy_file_path, output_filename):
        frames = np.load(npy_file_path)
        height, width = frames.shape[1], frames.shape[2]
        video_path = os.path.join(self.output_folder, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))

        for frame in frames:
            # Convert frame to BGR if it is grayscale for video compatibility
            if len(frame.shape) < 3 or frame.shape[2] != 3:
                frame = cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_GRAY2BGR)
            video.write(frame)

        video.release()
        print(f"Video saved at {video_path}")

class FrameDataIntegrator:
    def __init__(self, folder_path, positions_csv, frames_npy):
        self.folder_path = folder_path
        self.positions_csv = os.path.join(folder_path, positions_csv)
        self.frames_npy = os.path.join(folder_path, frames_npy)
        self.frame_data = np.load(self.frames_npy)
        self.positions_data = pd.read_csv(self.positions_csv)

    def segment_frames_by_direction(self):
        positions = self.positions_data['axis_y_position_mm'].values
        directions = np.sign(np.diff(positions))
        change_indices = np.where(np.diff(directions) != 0)[0] + 2

        start_idx = 0
        segments = []

        save_dir = os.path.join(self.folder_path, 'segmented_frames')
        os.makedirs(save_dir, exist_ok=True)

        converter = NpyToVideoConverter(self.folder_path)

        for idx in change_indices:
            end_idx = idx
            segment = self.frame_data[start_idx:end_idx]
            segment_filename = f'segment_{start_idx}_{end_idx}.npy'
            np.save(os.path.join(save_dir, segment_filename), segment)
            converter.convert_npy_to_video(os.path.join(save_dir, segment_filename), f'segment_{start_idx}_{end_idx}.mp4')
            start_idx = end_idx

        if start_idx < len(self.frame_data):
            segment = self.frame_data[start_idx:]
            segment_filename = f'segment_{start_idx}_{len(self.frame_data)}.npy'
            np.save(os.path.join(save_dir, segment_filename), segment)
            converter.convert_npy_to_video(os.path.join(save_dir, segment_filename), f'segment_{start_idx}_{len(self.frame_data)}.mp4')

if __name__ == '__main__':
    integrator = FrameDataIntegrator('data', 'filtered_frame_data_with_positions.csv', 'frames_output.npy')
    integrator.segment_frames_by_direction()
