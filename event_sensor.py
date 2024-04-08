import os
import cv2
import numpy as np
import time
from dv import NetworkEventInput, NetworkFrameInput, AedatFile, LegacyAedatFile, Control

class EventSensor:
    def __init__(self):
        self.address = '127.0.0.1'
        self.port = int(os.getenv('DV_PORT', '7777'))
        self.recording_enabled = False  # Initially, recording is not active

    def start_recording(self):
        self.recording_enabled = True

    def stop_recording(self):
        self.recording_enabled = False

    def record_events(self, output_file_name, recording_duration_sec=None):
        # Ensure the 'data' directory exists
        data_directory = 'data'
        if not os.path.exists(data_directory):
            os.makedirs(data_directory, exist_ok=True)

        # Path to save the output file
        output_file_path = os.path.join(data_directory, f"{output_file_name}.csv")

        # Open the output file for writing
        with open(output_file_path, 'w') as file:
            # Write the CSV header
            file.write("timestamp,x,y,polarity\n")

            with NetworkEventInput(address=self.address, port=self.port) as event_input:
                start_time = time.time()
                for event in event_input:
                    if not self.recording_enabled:
                        break  # Stop recording if the flag is set to False

                    # Write each event's details to the file in CSV format
                    file.write(f"{event.timestamp},{event.x},{event.y},{event.polarity}\n")
                    
                    # Check if the recording duration has been reached, if specified
                    if recording_duration_sec is not None and time.time() - start_time > recording_duration_sec:
                        break

        print(f"Events recorded to {output_file_path}")

    def listen_events(self):
        with NetworkEventInput(address=self.address, port=self.port) as event_input:
            for event in event_input:
                print(event.timestamp)

    def display_frames(self):
        with NetworkFrameInput(address=self.address, port=self.port) as frame_input:
            for frame in frame_input:
                print(frame)
                cv2.imshow('Live Frame', frame.image)
                cv2.waitKey(1)

    def open_recording(self, aedat_file_path):
        # Ensure the directory exists before attempting to open the file
        directory = os.path.dirname(aedat_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Directory {directory} created.")
            return  # Return early since the file won't exist yet

        try:
            with AedatFile(aedat_file_path) as f:
                print(f.names)
                for frame in f['frames']:
                    print(frame.timestamp)
                    cv2.imshow('Recorded Frame', frame.image)
                    cv2.waitKey(1)
        except FileNotFoundError as e:
            print(f"File not found: {aedat_file_path}")

    def access_events_numpy(self, aedat_file_path):
        with AedatFile(aedat_file_path) as f:
            events = np.hstack([packet for packet in f['events'].numpy()])
            print(events.shape)

    def load_legacy_file(self, file_path):
        with LegacyAedatFile(file_path) as f:
            for event in f:
                print(event.timestamp)

    def manage_dv_config(self, path, attribute, value_type, action='get', value=None):
        ctrl = Control(address=self.address, port=4040)
        if action == 'get':
            return ctrl.get(path, attribute, value_type)
        elif action == 'put':
            ctrl.put(path, attribute, value_type, value)
            print(f"Updated {path}/{attribute} to {value}")

# Example usage
sensor = EventSensor()
sensor.start_recording()  # Turn recording ON
sensor.record_events('events_output', 10)  # Optionally specify recording duration, or remove for indefinite recording

# sensor.listen_events()
# sensor.display_frames()
# sensor.open_recording('data/event.aedat')
# sensor.access_events_numpy('data/event.aedat')
# sensor.load_legacy_file('data/myFile.aedat')
# print(sensor.manage_dv_config('/mainloop/output_file/', 'file', 'string'))
