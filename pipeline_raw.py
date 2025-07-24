import sys
sys.path.append('/usr/lib/python3/dist-packages')

import argparse
import os
import torch
import numpy as np
from datetime import datetime, timedelta
from metavision_core.event_io import EventsIterator
from autocorrelation import determine_periphery
from block_visualizer import BlockVisualizer
import matplotlib.pyplot as plt
import subprocess
from integer_shifter import TensorShifter
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline from RAW to PT file (direct processing).')
    parser.add_argument('raw_path', help='Path to the input RAW file.')
    parser.add_argument('--output_dir', default=None, help='Directory to save the output files. Defaults to the input file\'s directory.')
    parser.add_argument('--start_ts', type=int, default=0, help='Start time to begin processing data in microseconds.')
    parser.add_argument('--max_duration', type=int, default=60000000, help='Maximum duration for processing data in microseconds.')
    parser.add_argument('--accumulation_time', type=int, default=10, help='Accumulation time in milliseconds for initial processing.')
    parser.add_argument('--height', type=int, default=None, help='Height of the event frame.')
    parser.add_argument('--width', type=int, default=None, help='Width of the event frame.')
    parser.add_argument('--model', choices=['D', 'X', 'E'], help='Specify camera model: D for Davis, X for Dvxplorer, E for Event sensor model E.')
    parser.add_argument('--force', action='store_true', help='Force override of existing files.')
    parser.add_argument('--save_frames', action='store_true', help='Save frames as images.')
    parser.add_argument('--alpha_scalar', type=float, default=0.7, help='Alpha scalar to adjust overall transparency.')
    parser.add_argument('--mean_start', type=int, default=2, help='Start index to mean over scanning pass.')
    parser.add_argument('--mean_end', type=int, default=3, help='End index to mean over scanning pass.')
    parser.add_argument('--auto_shift', action='store_true', help='Automatically determine the optimal shift value.')
    parser.add_argument('--min_shift', type=int, default=None, help='Minimum shift value to test for auto shift.')
    parser.add_argument('--max_shift', type=int, default=None, help='Maximum shift value to test for auto shift.')
    parser.add_argument('--sample_rate', type=int, default=10, help='Sampling rate to downsample the tensor before shifting.')
    parser.add_argument('--reverse', action='store_true', help='Whether to reverse the shift direction.')
    parser.add_argument('--shift', type=int, default=824, help='Shift value for the tensor transformation.')
    parser.add_argument('--n_acc', type=int, default=1, help='Number of frames to accumulate into one.')
    return parser.parse_args()

class DirectEventDataAccumulator:
    def __init__(self, 
        raw_path, start_ts, max_duration, output_folder, 
        accumulation_time=1, 
        height=None, width=None, 
        model=None, use_cache=False, force=False,
        reverse_time=False,
        reverse_polarity=False,
        description=None
    ):
        self.raw_path = raw_path
        self.start_ts = start_ts
        self.max_duration = max_duration
        self.output_folder = output_folder or os.path.dirname(raw_path)
        self.accumulation_time = accumulation_time  # in milliseconds
        os.makedirs(self.output_folder, exist_ok=True)

        raw_basename = os.path.basename(raw_path).replace('.raw', '')

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

        if description is not None:
            self.cache_path = os.path.join(
                self.output_folder, 
                f'{raw_basename}_frames_{description}_{self.height}_{self.width}_start_{start_ts}_dur_{max_duration}_acc_{self.accumulation_time}ms.pt'
            )
        else:
            self.cache_path = os.path.join(
                self.output_folder, 
                f'{raw_basename}_frames_{self.height}_{self.width}_start_{start_ts}_dur_{max_duration}_acc_{self.accumulation_time}ms.pt'
            )

        if os.path.exists(self.cache_path) and use_cache and not force:
            print("Loading frames from cache.")
            self.frames = torch.load(self.cache_path)
        else:
            self.load_and_process_data()
            if reverse_time:
                self.frames = torch.flip(self.frames, dims=[0])
            if reverse_polarity:
                self.frames = self.frames * -1
            torch.save(self.frames.cpu(), self.cache_path)
            print(f"Saved frames to {self.cache_path}.")

    def set_dynamic_dimensions(self):
        """Get dimensions from the raw file"""
        mv_iterator = EventsIterator(self.raw_path, delta_t=1000000, start_ts=0, max_duration=1000000)
        self.height, self.width = mv_iterator.get_size()
        print(f"Dynamic dimensions - Height: {self.height}, Width: {self.width}")

    def load_and_process_data(self):
        print(f"Loading events from raw file: {self.raw_path}")
        print(f"Start: {self.start_ts}μs, Duration: {self.max_duration}μs")
        
        # Create iterator for the specified time range
        mv_iterator = EventsIterator(
            input_path=self.raw_path,
            delta_t=1000000,  # Read in 1s chunks
            start_ts=self.start_ts,
            max_duration=self.max_duration
        )
        
        # If dimensions not set, get them from the file
        if not hasattr(self, 'height') or not hasattr(self, 'width'):
            self.height, self.width = mv_iterator.get_size()

        # Collect all events
        all_events = []
        print("Reading events...")
        for events in tqdm(mv_iterator, desc="Loading event chunks"):
            if events.size > 0:
                all_events.append(events)
        
        if not all_events:
            print("No events found!")
            self.frames = torch.zeros((0, self.height, self.width), dtype=torch.float32)
            return

        # Concatenate all events
        all_events = np.concatenate(all_events)
        print(f"Loaded {len(all_events):,} events")
        
        # Extract event data
        x = all_events['x']
        y = all_events['y']
        t = all_events['t']
        p = all_events['p']
        
        # Convert timestamps to relative time in milliseconds
        t_start = t[0]
        t_rel_ms = (t - t_start) / 1000.0  # Convert to milliseconds
        
        print(f"Time range: 0 - {t_rel_ms[-1]:.2f} ms")
        
        # Calculate number of time bins
        total_time_ms = t_rel_ms[-1]
        num_bins = int(total_time_ms / self.accumulation_time) + 1
        
        # Create frames tensor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.frames = torch.zeros((num_bins, self.height, self.width), dtype=torch.float32, device=device)
        
        # Convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.int64, device=device)
        y_tensor = torch.tensor(y, dtype=torch.int64, device=device)
        t_bins = torch.tensor((t_rel_ms / self.accumulation_time).astype(int), dtype=torch.int64, device=device)
        p_tensor = torch.tensor(p * 2 - 1, dtype=torch.float32, device=device)  # Convert 0,1 to -1,1
        
        # Accumulate events into frames
        print("Accumulating events into frames...")
        valid_mask = (t_bins < num_bins) & (x_tensor < self.width) & (y_tensor < self.height)
        
        if valid_mask.sum() > 0:
            self.frames.index_put_(
                (t_bins[valid_mask], y_tensor[valid_mask], x_tensor[valid_mask]), 
                p_tensor[valid_mask], 
                accumulate=True
            )
        
        print(f"Created frames tensor with shape: {self.frames.shape}")
        print(f"Frames value range: {self.frames.min():.3f} to {self.frames.max():.3f}")

def split_raw_to_segments(args, start_time, duration, interval):
    """Split raw file processing into segments"""
    output_paths = []
    reverse = False
    
    for i, start_relative in enumerate(range(0, int(duration), int(interval))):
        if i % 2 == 1:
            reverse = True
        else:
            reverse = False

        direction = ["forward", "backward"][reverse]
        segment_start = start_time + start_relative
        
        print(f"Processing segment {i+1} ({direction}): start={segment_start}μs, duration={interval}μs")
        
        data_processor = DirectEventDataAccumulator(
            raw_path=args.raw_path,
            start_ts=segment_start,
            max_duration=interval,
            output_folder=args.output_dir,
            accumulation_time=1,  # 1ms for fine segments
            height=args.height,
            width=args.width,
            model=args.model,
            force=args.force,
            use_cache=True,
            reverse_time=reverse,
            reverse_polarity=reverse,
            description=f"segment_{i+1}_{direction}"
        )
        
        output_paths.append(data_processor.cache_path)
    
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

    # Process initial data to determine period and periphery
    print("=== Initial processing to determine scanning parameters ===")
    data_processor = DirectEventDataAccumulator(
        raw_path=args.raw_path,
        start_ts=args.start_ts,
        max_duration=args.max_duration,
        output_folder=args.output_dir,
        accumulation_time=args.accumulation_time,
        height=args.height,
        width=args.width,
        model=args.model,
        force=args.force,
        use_cache=True,
    )

    pt_path_10ms = data_processor.cache_path
    raw_length = data_processor.frames.shape[0]

    print("=== Determining scanning period and periphery ===")
    prelude, aftermath, period = determine_periphery(pt_path_10ms, 10, device=device)
    print(f"Prelude: {prelude}, Aftermath: {aftermath}, Period: {period}")

    # Visualize initial tensor
    visualize_tensor(pt_path_10ms, args.output_dir)

    # Calculate timing for fine segments
    prelude_time = prelude * args.accumulation_time * 1000  # Convert to microseconds
    aftermath_time = aftermath * args.accumulation_time * 1000
    total_duration = period * args.accumulation_time * 3 * 1000  # 3 periods
    intervals = period * args.accumulation_time * 1000 // 2  # Half period intervals

    print(f"Prelude time: {prelude_time}μs")
    print(f"Total duration: {total_duration}μs") 
    print(f"Interval: {intervals}μs")

    # Process fine segments
    print("=== Processing fine segments ===")
    segment_start_time = args.start_ts + prelude_time
    small_pt_paths = split_raw_to_segments(args, segment_start_time, total_duration, intervals)

    # Visualize each segment
    for pt_path in small_pt_paths:
        visualize_tensor(pt_path, args.output_dir)

    # Create mean tensor
    print("=== Creating mean tensor ===")
    mean_tensor_output_path = os.path.join(args.output_dir, "mean_tensor.pt")
    mean_tensors(small_pt_paths, mean_tensor_output_path, start=mean_start, end=mean_end)
    visualize_tensor(mean_tensor_output_path, args.output_dir)

    # Determine optimal shift
    print("=== Determining optimal shift ===")
    if args.auto_shift:
        min_shift = args.min_shift if args.min_shift is not None else (600 if not args.reverse else -900)
        max_shift = args.max_shift if args.max_shift is not None else (900 if not args.reverse else -600)
        shift = determine_optimal_shift(mean_tensor_output_path, min_shift, max_shift, args.reverse, args.sample_rate)
    else:
        shift = args.shift

    print(f"Using shift value: {shift}")

    # Run background removal
    print("=== Running background removal ===")
    cmd = [
        'python', 'remove_background.py', mean_tensor_output_path,
        '--shift', str(shift),
        '--n_acc', str(args.n_acc),
        '--sample_rate', "1"
    ]

    if args.reverse:
        cmd.append('--reverse')

    if args.force:
        cmd.append('--force')

    if args.save_frames:
        cmd.append('--save_frames')

    subprocess.run(cmd)
    print("=== Pipeline completed ===")

if __name__ == '__main__':
    main()