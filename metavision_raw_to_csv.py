import sys
print(sys.path)
# Add /usr/lib/python3/dist-packages/ to PYTHONPATH if the output of print(sys.path) does not mention it.
sys.path.append("/usr/lib/python3/dist-packages/")

import os
import argparse
from datetime import datetime, timedelta
from metavision_core.event_io import EventsIterator
from tqdm import tqdm

def parse_time_argument(arg):
    """Parse time argument with units."""
    units = {"us": 1, "ms": 1000, "s": 1000000}
    if arg.isdigit():
        return int(arg) * units["s"]
    else:
        for unit in units:
            if arg.endswith(unit):
                return int(float(arg[:-len(unit)]) * units[unit])
    raise argparse.ArgumentTypeError(f"Invalid time format: {arg}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert Metavision RAW or DAT to CSV format.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-file', dest='input_path', required=True,
                        help="Path to input RAW or DAT file.")
    parser.add_argument('-o', '--output-dir', default=None,
                        help="Directory to save the output CSV file. Defaults to the input file's directory.")
    parser.add_argument('-s', '--start-ts', type=parse_time_argument, default="30s",
                        help="Start time to begin processing data. Can be specified in us, ms, or s (default: s).")
    parser.add_argument('-d', '--max-duration', type=parse_time_argument, default="60s",
                        help="Maximum duration for processing data. Can be specified in us, ms, or s (default: s).")
    parser.add_argument('--delta-t', type=parse_time_argument, default="1s",
                        help="Duration of served event slice. Can be specified in us, ms, or s (default: s).")
    parser.add_argument('--start-datetime', type=str, default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        help="Start datetime for timestamps in HH:MM:SS.microsecond format. Defaults to current datetime.")
    return parser.parse_args()

def main(args):
    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Construct the output filename with all the info in args
    output_filename = (
        f"{os.path.basename(args.input_path)[:-4]}"
        f"_start_ts_{args.start_ts}"
        f"_max_duration_{args.max_duration}"
        f"_delta_t_{args.delta_t}"
        # f"_start_datetime_{args.start_datetime.replace(' ', '_')}"
        ".csv"
    )
    output_file = os.path.join(args.output_dir, output_filename)

    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=args.delta_t,
                                 start_ts=args.start_ts, max_duration=args.max_duration)

    # Parse the default start datetime for converting timestamps
    start_datetime = datetime.strptime(args.start_datetime, "%Y-%m-%d %H:%M:%S")

    with open(output_file, 'w') as csv_file:
        # Write header
        csv_file.write("event_timestamp,x,y,polarity\n")
        for evs in tqdm(mv_iterator, total=args.max_duration // args.delta_t):
            for ev in evs:
                # Calculate event time and format it
                event_time = start_datetime + timedelta(microseconds=int(ev['t']))
                formatted_time = event_time.strftime('%H:%M:%S.%f')  # Trim to microseconds precision
                csv_file.write(f"{formatted_time},{ev['x']},{ev['y']},{ev['p']}\n")

    print(f"File saved to {output_file}. ")

if __name__ == "__main__":
    args = parse_args()
    main(args)
