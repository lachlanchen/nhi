import pandas as pd
import torch
import argparse
import os
import datetime

device = 'cpu'

class EventDataAccumulator:
    def __init__(self, csv_path, output_folder, accumulation_time=1, height=None, width=None, model=None, use_cache=False):
        self.csv_path = csv_path
        self.output_folder = output_folder
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
        if os.path.exists(self.cache_path) and use_cache:
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

        self.frames = torch.zeros((len(unique_intervals), self.height, self.width), dtype=torch.int8, device=device)
        x = torch.tensor(df['x'].values, dtype=torch.int64, device=device)
        y = torch.tensor(df['y'].values, dtype=torch.int64, device=device)
        polarity = torch.tensor(df['polarity'].map({True: 1, False: -1}).values, dtype=torch.int8, device=device)
        self.frames.index_put_((indices, y, x), polarity, accumulate=True)

    def save_frames_as_images(self):
        for i, frame in enumerate(self.frames):
            plt.figure(figsize=(8, 6))
            plt.imshow(frame.cpu().numpy(), cmap='gray', extent=[0, self.width, self.height, 0])
            plt.axis('off')
            output_path = os.path.join(self.output_folder, f'frame_{i:04d}.png')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        print(f"Saved frames as images in {self.output_folder} directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process event data to frame-like blocks.')
    parser.add_argument('csv_path', help='Path to the CSV file containing event data.')
    parser.add_argument('-output', dest='output_folder', required=True, help='Output folder for results.')
    parser.add_argument('-accumulation', type=int, default=1, help='Accumulation time in milliseconds (default: 1).')
    parser.add_argument('-size', nargs=2, type=int, help='Specify video width and height as W H.')
    parser.add_argument('-model', choices=['D', 'X', 'E'], help='Specify camera model: D for Davis, X for Dvxplorer, E for Event sensor model E.')

    args = parser.parse_args()

    height, width = None, None
    if args.size:
        height, width = args.size

    data_processor = EventDataAccumulator(args.csv_path, args.output_folder, args.accumulation, height, width, args.model)
    data_processor.save_frames_as_images()

