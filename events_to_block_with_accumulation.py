import pandas as pd
import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import subprocess

device = 'cpu'

class EventDataAccumulator:
    def __init__(self, csv_path, output_folder, accumulation_time=1, height=None, width=None, model=None, use_cache=False, force=False):
        self.csv_path = csv_path
        self.output_folder = output_folder or os.path.dirname(csv_path)
        self.accumulation_time = accumulation_time  # in milliseconds
        os.makedirs(self.output_folder, exist_ok=True)

        if model == 'D':
            self.height, self.width = 260, 346  # Davis
        elif model == 'X':
            self.height, self.width = 480, 640  # Dvxplorer
        elif model == 'E':
            self.height, self.width = 720, 1280  # Event sensor model E
        elif height is not None and width is not None:
            self.height, self.width = height, width
        else:
            self.set_dynamic_dimensions()

        self.cache_path = os.path.join(self.output_folder, f'frames_{self.height}_{self.width}.pt')
        if os.path.exists(self.cache_path) and use_cache and not force:
            print("Loading frames from cache.")
            self.frames = torch.load(self.cache_path).to(device)
        else:
            self.load_and_process_data()
            torch.save(self.frames.cpu(), self.cache_path)
            print(f"Saved frames to {self.cache_path}.")

    def set_dynamic_dimensions(self):
        df = pd.read_csv(self.csv_path)
        self.height = df['y'].max() + 1
        self.width = df['x'].max() + 1

        print("Height (y): ", self.height)
        print("Width (x): ", self.width)

    def load_and_process_data(self):
        df = pd.read_csv(self.csv_path)
        timestamps = pd.to_datetime(df['event_timestamp'], format='%H:%M:%S.%f')
        df['event_timestamp'] = (timestamps - timestamps.min()).dt.total_seconds() * 1000  # Convert to milliseconds

        accumulation_intervals = (df['event_timestamp'] // self.accumulation_time).astype(int)
        unique_intervals, indices = torch.unique(torch.tensor(accumulation_intervals.values), return_inverse=True)
        indices = indices.to(device)

        self.frames = torch.zeros((len(unique_intervals), self.height, self.width), dtype=torch.float32, device=device)
        x = torch.tensor(df['x'].values, dtype=torch.int64, device=device)
        y = torch.tensor(df['y'].values, dtype=torch.int64, device=device)
        polarity = torch.tensor(df['polarity'].map({0: -1, 1: 1}).values, dtype=torch.float32, device=device)
        self.frames.index_put_((indices, y, x), polarity, accumulate=True)

        print("self.frames.max(): ", self.frames.max())
        print("self.frames.min(): ", self.frames.min())

    def save_frames_as_images(self, alpha_scalar=0.7):
        image_folder = os.path.join(self.output_folder, 'events_frames')
        os.makedirs(image_folder, exist_ok=True)

        max_val = torch.max(torch.abs(self.frames)).item()

        for i, frame in enumerate(tqdm(self.frames, desc="Saving frames as images")):
            plt.figure(figsize=(8, 6))
            plt.axis('off')

            frame_np = frame.cpu().numpy()

            # Extract positive and negative events
            pos_y, pos_x = np.where(frame_np > 0)
            neg_y, neg_x = np.where(frame_np < 0)
            pos_vals = frame_np[pos_y, pos_x]
            neg_vals = frame_np[neg_y, neg_x]

            # Check if there are any positive or negative values to plot
            if len(pos_vals) > 0:
                pos_alpha = np.clip(np.abs(pos_vals) / max_val, 0, 1) * alpha_scalar
                plt.scatter(pos_x, pos_y, color='red', alpha=pos_alpha, s=1)
            if len(neg_vals) > 0:
                neg_alpha = np.clip(np.abs(neg_vals) / max_val, 0, 1) * alpha_scalar
                plt.scatter(neg_x, neg_y, color='blue', alpha=neg_alpha, s=1)

            output_path = os.path.join(image_folder, f'frame_{i:04d}.png')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        print(f"Saved frames as images in {image_folder} directory.")

    def frames_to_video(self, folder='events_frames', output_video='frames_video_with_ffmpeg.mp4', fps=60):
        folder_path = os.path.join(self.output_folder, folder)
        output_path = os.path.join(self.output_folder, output_video)
        command = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(folder_path, 'frame_%04d.png'),
            '-vf', 'scale=800:600',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        subprocess.run(command, check=True)
        print(f"Compiled frames into video: {output_path}")

    def frames_to_video_with_filename(self, folder='events_frames', output_video='frames_video_with_ffmpeg.mp4', fps=30):
        folder_path = os.path.join(self.output_folder, folder)
        output_path = os.path.join(self.output_folder, output_video)
        command = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(folder_path, 'frame_%04d.png'),
            '-vf', f'scale=800:600,drawtext=text=\'{output_video}\':fontcolor=white:fontsize=24:x=10:y=10',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        subprocess.run(command, check=True)
        print(f"Compiled frames into video: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process event data to frame-like blocks.')
    parser.add_argument('csv_path', help='Path to the CSV file containing event data.')
    parser.add_argument('-output', dest='output_folder', default=None, help='Output folder for results. Defaults to the input folder if not specified.')
    parser.add_argument('-accumulation', type=int, default=1, help='Accumulation time in milliseconds (default: 1).')
    parser.add_argument('-size', nargs=2, type=int, help='Specify video width and height as W H.')
    parser.add_argument('-model', choices=['D', 'X', 'E'], help='Specify camera model: D for Davis, X for Dvxplorer, E for Event sensor model E.')
    parser.add_argument('--force', action='store_true', help='Force override of existing files.')
    parser.add_argument('--save_frames', action='store_true', help='Save frames as images.')
    parser.add_argument('--alpha_scalar', type=float, default=0.7, help='Alpha scalar to adjust overall transparency (default: 0.7).')

    args = parser.parse_args()

    height, width = None, None
    if args.size:
        height, width = args.size

    data_processor = EventDataAccumulator(args.csv_path, args.output_folder, args.accumulation, height, width, args.model, force=args.force)

    if args.save_frames:
        # data_processor.save_frames_as_images(alpha_scalar=args.alpha_scalar)
        # data_processor.frames_to_video_with_filename()
        pass        

