import argparse
import numpy as np
import cv2
from metavision_core.event_io import EventsIterator
from metavision_sdk_base import EventCD
from metavision_sdk_ui import Window, EventLoop


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Metavision EVK5 Sensor Handling.")
    parser.add_argument('-i', '--input-event-file', default='', help="Path to input event file or camera serial number.")
    return parser.parse_args()

class EVK5Sensor:
    def __init__(self, input_path=''):
        """Initialize the sensor with either a file path or a live camera."""
        self.iterator = EventsIterator(input_path=input_path, delta_t=1000)
        self.running = False

    def start_recording(self):
        """Start recording events."""
        self.running = True

    def stop_recording(self):
        """Stop recording events."""
        self.running = False

    def record_events(self, output_file_name):
        """Record events to a CSV file."""
        recorded = False

        with open(output_file_name, 'w') as file:
            file.write("timestamp,x,y,polarity\n")
            for events in self.iterator:



                if not self.running:
                    break

                # evs = events
                # print("----- New event buffer! -----")
                # if evs.size == 0:
                #     print("The current event buffer is empty.")
                # else:
                #     min_t = evs['t'][0]   # Get the timestamp of the first event of this callback
                #     max_t = evs['t'][-1]  # Get the timestamp of the last event of this callback
                #     global_max_t = max_t  # Events are ordered by timestamp, so the current last event has the highest timestamp

                #     counter = evs.size  # Local counter
                #     # global_counter += counter  # Increase global counter

                #     print(f"There were {counter} events in this event buffer.")
                #     # print(f"There were {global_counter} total events up to now.")
                #     print(f"The current event buffer included events from {min_t} to {max_t} microseconds.")


                for event in events:
                    print(event)
                    x, y, polarity, timestamp = event
                    file.write(f"{timestamp},{x},{y},{polarity}\n")

                    recorded = True
                    break
                    # break

                if recorded:
                    break
        print(f"Events recorded to {output_file_name}")

    def display_events(self):
        """Display events using OpenCV."""
        window_name = "EVK5 Event Display"
        cv2.namedWindow(window_name)
        while self.running:
            EventLoop.poll_and_dispatch()  # Necessary to keep the window responsive
            events = next(self.iterator)
            frame = np.zeros((self.iterator.height, self.iterator.width, 3), dtype=np.uint8)
            for event in events:
                color = (255, 255, 255) if event.p else (0, 0, 255)
                frame[event.y, event.x] = color
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_recording()

    def process_events(self):
        """Process events and print basic statistics."""
        global_counter = 0
        global_max_t = 0
        for events in self.iterator:
            if not self.running:
                break
            min_t = events['t'][0]  # Timestamp of the first event
            max_t = events['t'][-1]  # Timestamp of the last event
            global_max_t = max_t
            counter = len(events)
            global_counter += counter
            print(f"Processed {counter} events, total {global_counter}. Range: {min_t} to {max_t}")
        print(f"Total events: {global_counter}")

def main():
    args = parse_args()
    sensor = EVK5Sensor(input_path=args.input_event_file)
    sensor.start_recording()
    # Choose one of the following based on your need
    sensor.record_events('output_events.csv')
    # sensor.display_events()
    # sensor.process_events()
    sensor.stop_recording()

if __name__ == "__main__":
    main()
