import os
import csv
import argparse
from metavision_core.event_io import EventsIterator
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser(description="Process events from EVK5 sensor.")
    parser.add_argument('-i', '--input', type=str, default='', help='Input path for the event data source.')
    parser.add_argument('-o', '--output', type=str, default='event_output.csv', help='Output CSV file name.')
    return parser.parse_args()

class EventSensorEVK5:
    def __init__(self, input_path=''):
        self.input_path = input_path
        self.running = False
        self.iterator = EventsIterator(input_path=self.input_path, delta_t=1000)

    def start_recording(self):
        self.running = True

    def stop_recording(self):
        self.running = False

    def record_events(self, output_file_name):
        """Record events to a CSV file."""
        recorded = False

        with open(output_file_name, 'w') as file:
            file.write("timestamp,x,y,polarity\n")
            for events in self.iterator:
                if not self.running:
                    break

                if len(events) > 0:
                    pprint(events)

                for event in events:
                    # Assuming each event is a tuple structured as (x, y, polarity, timestamp)
                    x, y, polarity, timestamp = event
                    file.write(f"{timestamp},{x},{y},{polarity}\n")
                    print(f"Recorded event: Timestamp={timestamp}, X={x}, Y={y}, Polarity={polarity}")

                    recorded = True
                    break  # Breaking after the first event for demonstration

                if recorded:
                    break  # Stop after processing the first batch of events

        print(f"Events recorded to {output_file_name}")

    def start_event_processing(self, output_file_name):
        self.start_recording()
        self.record_events(output_file_name)
        self.stop_recording()

if __name__ == "__main__":
    args = parse_args()
    sensor = EventSensorEVK5(input_path=args.input)
    sensor.start_event_processing(args.output)
