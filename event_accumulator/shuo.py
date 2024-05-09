import numpy as np
import pandas as pd
import os
import cv2
import argparse
from tqdm import tqdm

class EventProcessor:
    def __init__(self, csv_path, sensor_size=(260, 346)):  # Default to DAVIS 346x260
        self.csv_path = csv_path
        self.sensor_size = sensor_size
        self.data = self.load_data()
        self.output_directory = 'output_frames'
        self.ensure_directory(self.output_directory)

    def load_data(self):
        """Load event data from a CSV file."""
        data = pd.read_csv(self.csv_path)
        print(f"Data loaded from {self.csv_path}")
        return data

    def ensure_directory(self, path):
        """Ensure directory exists."""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory {path}")

    def process_events_to_frames(self, frame_interval=1000):
        """Process events and save to frames."""
        frame_list = []
        for index, event in self.data.iterrows():
            frame_path = os.path.join(self.output_directory, f"frame_{index // frame_interval}.jpg")
            if index % frame_interval == 0:
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                else:
                    frame = np.zeros(self.sensor_size, dtype=np.uint8)
            x, y, polarity = int(event['x']), int(event['y']), int(event['polarity'])
            frame[y, x] = np.clip(frame[y, x] + (1 if polarity else -1), 0, 255)
            if index % frame_interval == frame_interval - 1:
                cv2.imwrite(frame_path, frame)
                frame_list.append(frame)
        return frame_list

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process events to frames with optional sensor size.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing event data.")
    parser.add_argument("-s", "--size", type=str, help="Sensor size as H,W", default=None)
    parser.add_argument("-m", "--model", choices=['D', 'X'], help="Predefined model sizes: D for 346x260, X for 640x480", default=None)
    args = parser.parse_args()

    # Determine sensor size
    if args.size:
        sensor_size = tuple(map(int, args.size.split(',')))
    elif args.model == 'D':
        sensor_size = (260, 346)
    elif args.model == 'X':
        sensor_size = (480, 640)
    else:
        # Load CSV to determine max x and y if not specified
        data = pd.read_csv(args.csv_path)
        sensor_size = (data['y'].max() + 1, data['x'].max() + 1)

    return args.csv_path, sensor_size

if __name__ == "__main__":
    csv_path, sensor_size = parse_arguments()
    processor = EventProcessor(csv_path, sensor_size)
    frames = processor.process_events_to_frames()
    print(f"Processed {len(frames)} frames with sensor size {sensor_size}.")
