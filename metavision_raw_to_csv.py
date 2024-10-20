import sys
# Ensure the correct path for system libraries
print(sys.path)

import os
import argparse
from datetime import datetime, timedelta
from metavision_core.event_io import EventsIterator
from tqdm import tqdm

def parse_time_argument(arg):
    """Parse time argument with units."""
    units = {"us": 1, "ms": 1000, "s": 1000000}
    if isinstance(arg, int):
        return arg
    if arg.isdigit():
        return int(arg) * units["s"]
    else:
        for unit in units:
            if arg.endswith(unit):
                try:
                    return int(float(arg[:-len(unit)]) * units[unit])
                except ValueError:
                    break
    raise argparse.ArgumentTypeError(f"Invalid time format: {arg}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert Metavision RAW or DAT to CSV format.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-file', dest='input_path', required=True,
                        help="Path to input RAW or DAT file.")
    parser.add_argument('-o', '--output-dir', default=None,
                        help="Directory to save the output CSV file. Defaults to the input file's directory.")
    parser.add_argument('-s', '--start-ts', type=parse_time_argument, default="0s",
                        help="Start time to begin processing data. Can be specified in us, ms, or s (default: s).")
    parser.add_argument('-d', '--max-duration', type=parse_time_argument, default="60s",
                        help="Maximum duration for processing data. Can be specified in us, ms, or s (default: s).")
    parser.add_argument('--delta-t', type=parse_time_argument, default="1s",
                        help="Duration of served event slice. Can be specified in us, ms, or s (default: s).")
    parser.add_argument('--start-datetime', type=str, default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        help="Start datetime for timestamps in HH:MM:SS.microsecond format. Defaults to current datetime.")
    parser.add_argument('--force', action='store_true', help="Force override of existing files.")
    return parser.parse_args()

def save_start_time(start_time_file, start_datetime):
    """Save the start datetime to a text file."""
    with open(start_time_file, 'w') as f:
        f.write(start_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"))
    return start_time_file

def read_start_time(start_time_file):
    """Read the start datetime from a text file."""
    if os.path.exists(start_time_file):
        with open(start_time_file, 'r') as f:
            start_datetime_str = f.read().strip()
        return datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S.%f")
    return None

def calculate_time_difference_ignore_date(start_datetime, first_timestamp, delta_t=500):
    """Calculate the time difference only based on hour, minute, second, and microsecond part."""
    start_time = start_datetime.time()
    first_event_time = first_timestamp.time()
    time_diff_in_us = datetime.combine(datetime.min, first_event_time) - datetime.combine(datetime.min, start_time)

    print(f"Original start time: {start_time.strftime('%H:%M:%S.%f')}")
    print(f"First event time from CSV: {first_event_time.strftime('%H:%M:%S.%f')}")
    print(f"Time difference (HH:MM:SS): {time_diff_in_us} microseconds")

    time_diff_in_us = (time_diff_in_us // delta_t) * delta_t

    return int(time_diff_in_us.total_seconds() * 1_000_000)  # Convert to microseconds

def read_first_event_time_from_csv(csv_file_path):
    """Reads the first event timestamp from the CSV file."""
    with open(csv_file_path, 'r') as csv_file:
        csv_file.readline()  # Skip header
        first_line = csv_file.readline()  # Read the first data line
        first_timestamp_str = first_line.split(',')[0]
        first_timestamp = datetime.strptime(first_timestamp_str, '%H:%M:%S.%f')
    return first_timestamp

def handle_existing_file(csv_file_path, start_time_file):
    """Handles reading time difference if the CSV file already exists."""
    print(f"File {csv_file_path} already exists. Reading start time and calculating time difference.")
    
    # Read the start time from the saved text file
    start_datetime = read_start_time(start_time_file)
    if start_datetime is None:
        raise FileNotFoundError(f"Start time file not found for existing CSV file: {csv_file_path}")
    
    # Read the first event timestamp from the existing CSV
    first_timestamp = read_first_event_time_from_csv(csv_file_path)
    
    # Calculate the time difference in microseconds, ignoring the date
    time_diff_in_us = calculate_time_difference_ignore_date(start_datetime, first_timestamp)
    
    return csv_file_path, time_diff_in_us

def convert_raw_to_csv(
    input_path, output_dir=None, 
    start_ts=0, max_duration=60_000_000, delta_t=1_000_000, 
    start_datetime=None, force=False, description=None):
    
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_dir is None:
        output_dir = os.path.dirname(input_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if start_datetime is None:
        start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    description = f"_{description}" if description else ""
    # Construct the output file and start time file paths
    # csv_file_path = os.path.join(output_dir, f"{os.path.basename(input_path)[:-4]}_start_ts_{start_ts}_max_duration_{max_duration}_delta_t_{delta_t}.csv")
    # Construct the output filename with all the info in args, including description
    output_filename = (
        f"{os.path.basename(input_path)[:-4]}"
        f"{description}"
        f"_start_ts_{start_ts}"
        f"_max_duration_{max_duration}"
        f"_delta_t_{delta_t}"
        ".csv"
    )
    csv_file_path = os.path.join(output_dir, output_filename)
    start_time_file = os.path.join(output_dir, f"{os.path.basename(input_path)[:-4]}_start_time.txt")

    # Check if the CSV file already exists
    if os.path.exists(csv_file_path) and not force:
        return handle_existing_file(csv_file_path, start_time_file)

    # Start converting RAW to CSV
    mv_iterator = EventsIterator(
        input_path=input_path, 
        delta_t=delta_t,
        start_ts=start_ts, 
        max_duration=max_duration
    )

    # Parse the default start datetime for converting timestamps
    start_datetime = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")

    # Save the start datetime to a text file
    save_start_time(start_time_file, start_datetime)

    with open(csv_file_path, 'w') as csv_file:
        # Write header
        csv_file.write("event_timestamp,x,y,polarity\n")
        total_iterations = max_duration // delta_t
        for evs in tqdm(mv_iterator, total=total_iterations):
            for ev in evs:
                # Calculate event time and format it
                event_time = start_datetime + timedelta(microseconds=int(ev['t']))
                formatted_time = event_time.strftime('%H:%M:%S.%f')  # Trim to microseconds precision
                csv_file.write(f"{formatted_time},{ev['x']},{ev['y']},{ev['p']}\n")

    print(f"File saved to {csv_file_path}.")

    # Read the first event timestamp from the newly created CSV
    first_timestamp = read_first_event_time_from_csv(csv_file_path)

    # Calculate the time difference between the start_datetime and the first timestamp in the CSV, ignoring date
    time_diff_in_us = calculate_time_difference_ignore_date(start_datetime, first_timestamp)

    return csv_file_path, time_diff_in_us

def main():
    args = parse_args()
    csv_file_path, time_diff_us = convert_raw_to_csv(
        input_path=args.input_path,
        output_dir=args.output_dir,
        start_ts=args.start_ts,
        max_duration=args.max_duration,
        delta_t=args.delta_t,
        start_datetime=args.start_datetime,
        force=args.force
    )
    print(f"CSV saved to {csv_file_path}. Time difference (HH:MM:SS): {time_diff_us} microseconds")

if __name__ == "__main__":
    main()
