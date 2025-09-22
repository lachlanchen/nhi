import sys
sys.path.append('/usr/lib/python3/dist-packages')


import argparse
import os
import torch
import numpy as np
from datetime import datetime, timedelta
from metavision_raw_to_csv import convert_raw_to_csv, parse_time_argument
from events_to_block_with_accumulation import EventDataAccumulator
from autocorrelation import calculate_autocorrelation, find_top_peaks, find_period, calculate_periphery, load_tensor, determine_periphery
from block_visualizer import BlockVisualizer
import matplotlib.pyplot as plt
import subprocess
from integer_shifter import TensorShifter
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline from RAW to PT file.')
    parser.add_argument('raw_path', help='Path to the input RAW file.')
    parser.add_argument('--output_dir', default=None, help='Directory to save the output files. Defaults to the input file\'s directory.')
    parser.add_argument('--start_ts', type=parse_time_argument, default="0s", help='Start time to begin processing data. Can be specified in us, ms, or s.')
    parser.add_argument('--max_duration', type=parse_time_argument, default="60s", help='Maximum duration for processing data. Can be specified in us, ms, or s.')
    parser.add_argument('--delta_t', type=parse_time_argument, default="1s", help='Duration of served event slice. Can be specified in us, ms, or s.')
    parser.add_argument('--start_datetime', type=str, default=None, help='Start datetime for timestamps in HH:MM:SS.microsecond format. Defaults to current datetime.')
    parser.add_argument('--accumulation_time', type=int, default=10, help='Accumulation time in milliseconds for initial processing.')
    parser.add_argument('--height', type=int, default=None, help='Height of the event frame.')
    parser.add_argument('--width', type=int, default=None, help='Width of the event frame.')
    parser.add_argument('--model', choices=['D', 'X', 'E'], help='Specify camera model: D for Davis, X for Dvxplorer, E for Event sensor model E.')
    parser.add_argument('--force', action='store_true', help='Force override of existing files.')
    parser.add_argument('--save_frames', action='store_true', help='Save frames as images.')
    parser.add_argument('--alpha_scalar', type=float, default=0.7, help='Alpha scalar to adjust overall transparency.')
    parser.add_argument('--mean_start', type=int, default=2, help='Start index to mean over scanning pass. ')
    parser.add_argument('--mean_end', type=int, default=3, help='End index to mean over scanning pass. ')
    parser.add_argument('--auto_shift', action='store_true', help='Automatically determine the optimal shift value.')
    parser.add_argument('--min_shift', type=int, default=None, help='Minimum shift value to test for auto shift.')
    parser.add_argument('--max_shift', type=int, default=None, help='Maximum shift value to test for auto shift.')
    parser.add_argument('--sample_rate', type=int, default=10, help='Sampling rate to downsample the tensor before shifting.')
    parser.add_argument('--reverse', action='store_true', help='Whether to reverse the shift direction.')
    parser.add_argument('--shift', type=int, default=824, help='Shift value for the tensor transformation.')  # Default shift value
    parser.add_argument('--n_acc', type=int, default=1, help='Number of frames to accumulate into one.')
    return parser.parse_args()

def convert_time_to_microseconds(time_str):
    units = {"us": 1, "ms": 1000, "s": 1000000}
    for unit in units:
        if time_str.endswith(unit):
            return int(float(time_str[:-len(unit)]) * units[unit])
    return int(time_str) * units["s"]

def adjust_delta_t(start_ts, delta_t):
    # start_ts_us = convert_time_to_microseconds(start_ts)
    # delta_t_us = convert_time_to_microseconds(delta_t)
    start_ts_us = start_ts
    delta_t_us = delta_t

    while start_ts_us % delta_t_us != 0:
        delta_t_us -= 1

    return f"{delta_t_us}us"

def split_raw_to_csv(args, start_time, duration, interval, time_diff=0):
    start_time = start_time + args.start_ts + time_diff # convert_time_to_microseconds(args.start_ts)
    output_paths = []
    reverse = False
    for i, start_relative in enumerate(range(0, int(duration), int(interval))):
        # adjusted_delta_t = "1000us"
        adjusted_delta_t = 500
        if i % 2 == 1:
            reverse = True
        else:
            reverse = False

        direction = ["forward", "backward"][reverse]
        print("start_ts: ", start_time + start_relative)
        print("interval: ", interval)
        output_csv, _ = convert_raw_to_csv(
            input_path=args.raw_path,
            output_dir=args.output_dir,
            # start_ts=f"{start_time + start_relative}us",
            # max_duration=f"{interval}us",
            start_ts=start_time + start_relative,
            max_duration=interval,
            delta_t=adjusted_delta_t,
            start_datetime=args.start_datetime,
            description=f"segment_{i+1}_{direction}"
        )
        output_paths.append(output_csv)
    return output_paths

def visualize_tensor(tensor_path, output_dir, sample_rate=10, time_stretch=5):
    visualizer = BlockVisualizer(tensor_path, sample_rate=sample_rate)

    views = ["default", "vertical", "horizontal", "side", "r-side", "normal", "normal45", "lateral", "reverse"]

    base_name = os.path.splitext(os.path.basename(tensor_path))[0]
    save_folder = os.path.join(output_dir, base_name)
    os.makedirs(save_folder, exist_ok=True)

    for view in views:
        save_path = os.path.join(save_folder, f"{base_name}_{view}.png")
        if os.path.exists(save_path):
            print(f"Skipping existing visualization: {save_path}")
            continue
        visualizer.plot_scatter_tensor(view=view, plot=False, save=True, save_path=save_path, time_stretch=time_stretch)

def plot_height_projection(tensor, output_dir, tensor_name):
    projection = tensor.mean(dim=2).cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.imshow(projection, aspect='auto', cmap='viridis')
    plt.colorbar(label='Mean Intensity')
    plt.xlabel('Time')
    plt.ylabel('Height')
    plt.title(f'Projection on Height Axis - {tensor_name}')
    save_path = os.path.join(output_dir, f"{tensor_name}_height_projection.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved height projection plot to {save_path}")

def mean_tensors(tensor_paths, output_path, start=1, end=5):
    mean_tensor = None
    count = 0
    output_dir = os.path.dirname(output_path)
    for path in tensor_paths[start:end]:
        tensor = torch.load(path)
        plot_height_projection(tensor, output_dir, os.path.splitext(os.path.basename(path))[0])
        if mean_tensor is None:
            mean_tensor = torch.zeros_like(tensor, dtype=torch.float32)
        min_size = min(mean_tensor.shape[0], tensor.shape[0])
        mean_tensor[-min_size:] += tensor[-min_size:]
        count += 1

    mean_tensor /= count
    torch.save(mean_tensor, output_path)
    print(f"Saved mean tensor to {output_path}")
    plot_height_projection(mean_tensor, output_dir, "mean_tensor")

def calculate_std_for_shifts(tensor, min_shift, max_shift, reverse, sample_rate):
    std_values = []
    shift_values = list(range(min_shift, max_shift + 1))
    width = tensor.shape[2]
    tensor_shifter = TensorShifter(0, width // sample_rate, reverse)

    for shift in tqdm(shift_values):
        tensor_shifter.max_shift = shift
        sampled_tensor = tensor[:, ::sample_rate, ::sample_rate]
        shifted_tensor = tensor_shifter.apply_shift(sampled_tensor)
        std_val = torch.sum(torch.std(shifted_tensor, dim=[1, 2])).item()
        std_values.append(std_val)

    return shift_values, std_values

def optimal_shift(shift_values, std_values):
    min_std = min(std_values)
    min_shift = shift_values[std_values.index(min_std)]
    print(f"Minimum standard deviation: {min_std:.2f} occurs at shift: {min_shift}")
    return min_shift, min_std

def plot_std_vs_shift(shift_values, std_values, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(shift_values, std_values, marker='o', linestyle='-')
    min_std = min(std_values)
    min_shift = shift_values[std_values.index(min_std)]
    plt.scatter(min_shift, min_std, color='red')
    plt.text(min_shift, min_std, f'Min Std: {min_std:.2f} at Shift: {min_shift}', fontsize=12, color='red')
    plt.axvline(x=min_shift, color='red', linestyle='--')
    plt.xlabel('Shift Value')
    plt.ylabel('Sum of Standard Deviations of Shifted Frames')
    plt.title('Sum of Standard Deviations vs. Shift Value')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Figure saved to {output_path}")
    print(f"Minimum standard deviation: {min_std:.2f} occurs at shift: {min_shift}")

def determine_optimal_shift(tensor_path, min_shift, max_shift, reverse, sample_rate):
    tensor = torch.load(tensor_path)
    shift_values, std_values = calculate_std_for_shifts(tensor, min_shift, max_shift, reverse, sample_rate)
    dir_name = os.path.dirname(tensor_path)
    reverse_str = 'reversed' if reverse else 'normal'
    figure_filename = f"std_plot_{min_shift}_to_{max_shift}_{reverse_str}_sample{sample_rate}.png"
    output_path = os.path.join(dir_name, figure_filename)
    plot_std_vs_shift(shift_values, std_values, output_path)
    min_shift, _ = optimal_shift(shift_values, std_values)
    return min_shift

def main():
    args = parse_args()
    mean_start = args.mean_start
    mean_end = args.mean_end
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.raw_path)

    csv_path, time_diff = convert_raw_to_csv(
        input_path=args.raw_path,
        output_dir=args.output_dir,
        start_ts=args.start_ts,
        max_duration=args.max_duration,
        delta_t=args.delta_t,
        start_datetime=args.start_datetime
    )

    data_processor = EventDataAccumulator(
        csv_path,
        args.output_dir,
        accumulation_time=args.accumulation_time,
        height=args.height,
        width=args.width,
        model=args.model,
        force=args.force,
        use_cache=True,
    )

    pt_path_10ms = data_processor.cache_path
    raw_length = data_processor.frames.shape[0]

    prelude, aftermath, period = determine_periphery(pt_path_10ms, 10, device=device)
    print(f"Prelude: {prelude}, Aftermath: {aftermath}, Period: {period}")

    visualize_tensor(pt_path_10ms, args.output_dir)

    # period = 1250
    prelude_time = prelude * args.accumulation_time
    aftermath_time = aftermath * args.accumulation_time
    total_duration = period * args.accumulation_time * 3
    intervals = period * args.accumulation_time / 2

    print("prelude_time: ", prelude_time)
    print("total_duration: ", total_duration)
    print("intervals: ", intervals)

    small_csv_paths = split_raw_to_csv(args, prelude_time * 1000, total_duration * 1000, intervals * 1000, time_diff=time_diff)

    reverse = False
    pt_paths_1ms = []
    for i, csv_path in enumerate(small_csv_paths):
        if i % 2 == 1:
            reverse = True
        else:
            reverse = False
        
        data_processor = EventDataAccumulator(
            csv_path,
            args.output_dir,
            accumulation_time=1,
            height=args.height,
            width=args.width,
            model=args.model,
            force=args.force,
            use_cache=True,
            reverse_time=reverse,
            reverse_polarity=reverse,
        )
        pt_path_1ms = data_processor.cache_path
        pt_paths_1ms.append(pt_path_1ms)

        visualize_tensor(pt_path_1ms, args.output_dir)

    mean_tensor_output_path = os.path.join(args.output_dir, "mean_tensor.pt")
    mean_tensors(pt_paths_1ms, mean_tensor_output_path, start=mean_start, end=mean_end)
    visualize_tensor(mean_tensor_output_path, args.output_dir)

    # raise 

    if args.auto_shift:
        min_shift = args.min_shift if args.min_shift is not None else (600 if not args.reverse else -900)
        max_shift = args.max_shift if args.max_shift is not None else (900 if not args.reverse else -600)
        shift = determine_optimal_shift(mean_tensor_output_path, min_shift, max_shift, args.reverse, args.sample_rate)
    else:
        shift = args.shift

    print(f"Determined optimal shift: {shift}")

    cmd = [
        'python', 'remove_background.py', mean_tensor_output_path,
        '--shift', str(shift),
        '--n_acc', str(args.n_acc),
        # '--sample_rate', str(args.sample_rate)
        '--sample_rate', "1"
    ]

    if args.reverse:
        cmd.append('--reverse')

    # if args.output_dir:
    #     cmd.extend(['--output_dir', args.output_dir])

    if args.force:
        cmd.append('--force')

    if args.save_frames:
        cmd.append('--save_frames')

    subprocess.run(cmd)

if __name__ == '__main__':
    main()
