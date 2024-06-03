import argparse
import os
import torch
import numpy as np
from datetime import datetime, timedelta
from metavision_raw_to_csv import convert_raw_to_csv
from events_to_block_with_accumulation import EventDataAccumulator
from autocorrelation import calculate_autocorrelation, find_top_peaks, find_period, calculate_periphery, load_tensor
from autocorrelation import determine_prephery

def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline from RAW to PT file.')
    parser.add_argument('raw_path', help='Path to the input RAW file.')
    parser.add_argument('--output_dir', default=None, help='Directory to save the output files. Defaults to the input file\'s directory.')
    parser.add_argument('--start_ts', type=str, default="0s", help='Start time to begin processing data. Can be specified in us, ms, or s.')
    parser.add_argument('--max_duration', type=str, default="60s", help='Maximum duration for processing data. Can be specified in us, ms, or s.')
    parser.add_argument('--delta_t', type=str, default="1s", help='Duration of served event slice. Can be specified in us, ms, or s.')
    parser.add_argument('--start_datetime', type=str, default=None, help='Start datetime for timestamps in HH:MM:SS.microsecond format. Defaults to current datetime.')
    parser.add_argument('--accumulation_time', type=int, default=10, help='Accumulation time in milliseconds for initial processing.')
    parser.add_argument('--height', type=int, default=None, help='Height of the event frame.')
    parser.add_argument('--width', type=int, default=None, help='Width of the event frame.')
    parser.add_argument('--model', choices=['D', 'X', 'E'], help='Specify camera model: D for Davis, X for Dvxplorer, E for Event sensor model E.')
    parser.add_argument('--force', action='store_true', help='Force override of existing files.')
    parser.add_argument('--save_frames', action='store_true', help='Save frames as images.')
    parser.add_argument('--alpha_scalar', type=float, default=0.7, help='Alpha scalar to adjust overall transparency.')
    return parser.parse_args()

def split_raw_to_csv(args, start_time, duration, interval):
    output_paths = []
    for i in range(0, int(duration), int(interval)):
        output_csv = convert_raw_to_csv(
            input_path=args.raw_path,
            output_dir=args.output_dir,
            start_ts=f"{start_time + i}ms",
            max_duration=f"{interval}ms",
            delta_t=args.delta_t,
            start_datetime=args.start_datetime
        )
        output_paths.append(output_csv)
    return output_paths

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.raw_path)

    # Step 1: Convert RAW to CSV for the initial duration
    csv_path = convert_raw_to_csv(
        input_path=args.raw_path,
        output_dir=args.output_dir,
        start_ts=args.start_ts,
        max_duration=args.max_duration,
        delta_t=args.delta_t,
        start_datetime=args.start_datetime
    )

    # Step 2: Accumulate events into frames and save to PT file with 10 ms accumulation time
    data_processor = EventDataAccumulator(
        csv_path,
        args.output_dir,
        accumulation_time=args.accumulation_time,
        height=args.height,
        width=args.width,
        model=args.model,
        force=args.force
    )

    pt_path_10ms = data_processor.cache_path
    # pt_path_10ms = os.path.join(args.output_dir, f"{os.path.basename(csv_path).replace('.csv', '')}_10ms.pt")
    torch.save(data_processor.frames.cpu(), pt_path_10ms)

    prelude, aftermath, period = determine_prephery(pt_path_10ms, 10, device=device)
    print(f"Prelude: {prelude}, Aftermath: {aftermath}")

    # Convert frame indices to time in milliseconds
    prelude_time = prelude * args.accumulation_time
    aftermath_time = aftermath * args.accumulation_time
    total_duration = aftermath_time - prelude_time
    intervals = total_duration // 6

    # Step 5: Convert RAW to multiple small CSV files based on the calculated times
    small_csv_paths = split_raw_to_csv(args, prelude_time, total_duration, intervals)

    # Step 6: Convert each small CSV to PT files with 1 ms accumulation time
    for csv_path in small_csv_paths:
        data_processor = EventDataAccumulator(
            csv_path,
            args.output_dir,
            accumulation_time=1,  # 1 ms for finer accumulation
            height=args.height,
            width=args.width,
            model=args.model,
            force=args.force
        )
        pt_path_1ms = os.path.join(args.output_dir, f"{os.path.basename(csv_path).replace('.csv', '')}_1ms.pt")
        torch.save(data_processor.frames.cpu(), pt_path_1ms)

if __name__ == '__main__':
    main()
