import sys
print(sys.path)
# Add /usr/lib/python3/dist-packages/ to PYTHONPATH if the output of print(sys.path) does not mention it.
sys.path.append("/usr/lib/python3/dist-packages/")

import sys
import os
import argparse
from datetime import datetime, timedelta
from metavision_core.event_io import EventsIterator
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert Metavision RAW or DAT to CSV format.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-file', dest='input_path', required=True,
                        help="Path to input RAW or DAT file.")
    parser.add_argument('-o', '--output-dir', default=None,
                        help="Directory to save the output CSV file. Defaults to the input file's directory.")
    parser.add_argument('-s', '--start-ts', type=int, default=int(1e6 * 30),
                        help="Start time in microseconds to begin processing data.")
    parser.add_argument('-d', '--max-duration', type=int, default=int(1e6 * 60),
                        help="Maximum duration in microseconds.")
    parser.add_argument('--delta-t', type=int, default=1000000,
                        help="Duration of served event slice in microseconds.")
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

    output_file = os.path.join(args.output_dir, os.path.basename(args.input_path)[:-4] + ".csv")
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

if __name__ == "__main__":
    args = parse_args()
    main(args)

