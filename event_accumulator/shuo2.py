import numpy as np
import pandas as pd
import glob, os
from tqdm import tqdm
import argparse
from datetime import datetime

# Default Constants
DEFAULT_START_TIME = 0.86 * 1e6  # in microseconds
DEFAULT_TIME_WINDOW = 0.03 * 1e6  # in microseconds
DEFAULT_INTERVAL_TIME = 0.2 * 1e6  # in microseconds
DEFAULT_ROL_SIZE = 512
# DEFAULT_DATA_PATH = '../data/0512/'
# DEFAULT_FILE_NAME = '2.5mm'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process event data for correlations.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing event data.")
    parser.add_argument("-s", "--size", type=str, help="Sensor size as Height,Width", default="260,346")
    parser.add_argument("-m", "--model", choices=['D', 'X'], help="Predefined model sizes: D for 260x346, X for 480x640")
    args = parser.parse_args()

    # Determine sensor size
    if args.model == 'D':
        sensor_size = (260, 346)
    elif args.model == 'X':
        sensor_size = (480, 640)
    else:
        height, width = map(int, args.size.split(','))
        sensor_size = (height, width)
    
    return args.file_path, sensor_size

def load_data(file_path):
    """Load event data from a CSV file and convert timestamps."""
    data = pd.read_csv(file_path)
    data['event_timestamp'] = pd.to_datetime(data['event_timestamp']).view(np.int64) / 1e3
    return data

def form_bii(event_sel, roi):
    """Forms a Binary Image Indicator (BII) from event selection."""
    bii = np.zeros((roi[1]-roi[0], roi[3]-roi[2], 2), dtype=int)
    for event in event_sel:
        x_offset = event[2] - roi[2]
        y_offset = event[3] - roi[0]
        polarity_index = 0 if event[4] else 1
        bii[y_offset, x_offset, polarity_index] += 1
    return bii - np.mean(bii, axis=(0, 1), keepdims=True)

def crop_events(event_sel, start_time, interval_time, time_window):
    """Crop events within specified time window."""
    event_1 = event_sel[(event_sel[:, 0] >= start_time) & (event_sel[:, 0] < start_time + time_window)]
    event_2 = event_sel[(event_sel[:, 0] >= start_time + interval_time) & (event_sel[:, 0] < start_time + interval_time + time_window)]
    min_size = min(len(event_1), len(event_2))
    return event_1[:min_size], event_2[:min_size]

def corr2D(A, B):
    """Compute 2D correlation between two arrays."""
    return abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))))).real

if __name__ == "__main__":
    file_path, sensor_size = parse_arguments()
    data = load_data(file_path)
    print(f"Data loaded and timestamps converted for {os.path.basename(file_path)}")

    roi = [0, sensor_size[1], 0, sensor_size[0]]
    start_time = data['event_timestamp'].min() + DEFAULT_START_TIME
    time_window = DEFAULT_TIME_WINDOW
    interval_time = DEFAULT_INTERVAL_TIME

    data_crop = data[(data['x'] >= roi[2]) & (data['x'] < roi[1]) & (data['y'] >= roi[0]) & (data['y'] < roi[3])]
    event_a, event_b = crop_events(data_crop.to_numpy(), start_time, interval_time, time_window)
    bii_a = form_bii(event_a, roi)
    bii_b = form_bii(event_b, roi)
    corr = corr2D(bii_a[:, :, 0] + bii_a[:, :, 1], bii_b[:, :, 0] + bii_b[:, :, 1])
    corr[roi[1]//2, roi[0]//2] = 0

    max_corr = np.max(corr)
    indices = np.unravel_index(np.argmax(corr), corr.shape)
    dispx = roi[1]//2 - indices[1]
    dispy = roi[0]//2 - indices[0]
    print(f"Max correlation: {max_corr}\nDisplacement from correlation: (dx={dispx}, dy={dispy})")
