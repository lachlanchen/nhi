import os
import csv
import time
import argparse
from datetime import datetime, timedelta
import pytz
from metavision_core.event_io import EventsIterator
from event_sensor import timestamp_to_formatted_string_with_timezone

def parse_args():
    parser = argparse.ArgumentParser(description="Process events from EVK5 sensor.")
    parser.add_argument('-i', '--input', type=str, default='', help='Input path for the event data source.')
    parser.add_argument('-d', '--duration', type=int, default=None, help='Recording duration in seconds, if applicable.')
    parser.add_argument('-z', '--timezone', type=str, default='Asia/Hong_Kong', help='Timezone for timestamp conversion.')
    parser.add_argument('-o', '--output', type=str, default='event_output.csv', help='Output CSV file name.')
    return parser.parse_args()

# def timestamp_to_formatted_string_with_timezone(timestamp_microseconds, timezone='Asia/Hong_Kong'):
#     epoch_start = datetime(1970, 1, 1, tzinfo=pytz.utc)
#     time_since_epoch = timedelta(microseconds=timestamp_microseconds)
#     event_datetime_utc = epoch_start + time_since_epoch
#     timezone_obj = pytz.timezone(timezone)
#     event_datetime_tz = event_datetime_utc.astimezone(timezone_obj)
#     return event_datetime_tz.strftime("%H:%M:%S.%f")

class EventSensorEVK5:
    def __init__(self, input_path=''):
        self.input_path = input_path
        self.recording_enabled = False
        self.iterator = EventsIterator(input_path=self.input_path, delta_t=1000)

    def start_recording(self):
        self.recording_enabled = True

    def stop_recording(self):
        self.recording_enabled = False

    def record_events_with_auxiliary_data(self, output_file_name, recording_duration_sec=None, timezone='Asia/Hong_Kong'):
        events_data = []  # Memory storage for event data
        data_directory = 'data'
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)

        output_file_path = os.path.join(data_directory, output_file_name)
        start_time = time.time() 
        start_timestamp = int(start_time * 1e6)

        for events in self.iterator:
            if not self.recording_enabled:
                break

            system_timestamp_microseconds = int(time.time() * 1e6)
            system_timestamp_formatted = timestamp_to_formatted_string_with_timezone(system_timestamp_microseconds, timezone)
            
            for event in events:
                x, y, polarity, timestamp = event  # Assuming the event is a tuple (x, y, polarity, timestamp)
                event_timestamp_formatted = timestamp_to_formatted_string_with_timezone(timestamp+start_timestamp, timezone)

                events_data.append([event_timestamp_formatted, system_timestamp_formatted, x, y, polarity])

            if recording_duration_sec is not None and time.time() - start_time > recording_duration_sec:
                break

        # After recording, save all data to CSV
        with open(output_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["event_timestamp", "system_timestamp", "x", "y", "polarity"])
            writer.writerows(events_data)
        
        print(f"Events recorded to {output_file_path}")

    def record_events_and_frames_concurrently(self, event_file_name, frame_file_name, recording_duration_sec=None):
        self.record_events_with_auxiliary_data(event_file_name, recording_duration_sec=recording_duration_sec)

    def start_event_processing(self, output_file_name, recording_duration_sec, timezone):
        self.start_recording()
        self.record_events_with_auxiliary_data(output_file_name, recording_duration_sec, timezone)
        self.stop_recording()

if __name__ == "__main__":
    args = parse_args()
    sensor = EventSensorEVK5(input_path=args.input)
    sensor.start_event_processing(args.output, args.duration, args.timezone)
