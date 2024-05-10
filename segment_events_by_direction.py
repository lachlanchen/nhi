import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pprint import pprint
import argparse

class EventSegmenter:
    def __init__(self, base_folder):
        self.csv_path = os.path.join(base_folder, 'filtered_event_data_with_positions.csv')
        self.output_folder = os.path.join(base_folder, 'segmented_events')
        os.makedirs(self.output_folder, exist_ok=True)  # Ensure the output directory exists

    def load_data(self):
        # Load data from CSV
        return pd.read_csv(self.csv_path)

    def segment_by_direction(self):
        data = self.load_data()

        # Calculate the difference to identify position changes
        pos_diff = data['axis_y_position_mm'].diff()

        # Initialize variables
        last_known_direction = np.sign(pos_diff.iloc[1])
        change_indices = []
        directions = np.zeros(len(data))

        # Loop through differences to assign directions and detect changes
        for i in tqdm(range(1, len(pos_diff))):
            current_diff = pos_diff.iloc[i]
            if current_diff != 0:
                current_direction = np.sign(current_diff)
            else:
                current_direction = last_known_direction

            if current_direction != last_known_direction and current_direction != 0:
                change_indices.append(i)
                last_known_direction = current_direction
            
            directions[i] = current_direction

        # Collect segments based on detected changes in direction
        start_idx = 0
        segments = []

        for idx in change_indices:
            # idx = idx + 1
            segment = data.iloc[start_idx:idx]
            segment_filename = f'segment_{start_idx}_{idx-1}.csv'
            segment.to_csv(os.path.join(self.output_folder, segment_filename), index=False)
            segments.append(segment)
            start_idx = idx

        # Handle the last segment if any data remains
        if start_idx < len(data):
            segment = data.iloc[start_idx:]
            segment_filename = f'segment_{start_idx}_{len(data)-1}.csv'
            segment.to_csv(os.path.join(self.output_folder, segment_filename), index=False)
            segments.append(segment)

        return segments

# Usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment events by direction from a CSV file located in a specified folder')
    parser.add_argument('base_folder', type=str, help='Base folder name containing the CSV file and where segmented events will be stored')
    args = parser.parse_args()

    segmenter = EventSegmenter(args.base_folder)
    segments = segmenter.segment_by_direction()
    print(f"{len(segments)} segments created and saved in {args.base_folder}")
