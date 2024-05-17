import os
import cv2
import numpy as np
import time
from dv import NetworkEventInput, NetworkFrameInput, AedatFile, LegacyAedatFile, Control

from datetime import datetime, timedelta
import pytz
import csv
import threading

def timestamp_to_formatted_string_with_timezone(timestamp_microseconds, timezone='Asia/Hong_Kong'):
    # Assuming the timestamp is based on the UNIX epoch (January 1, 1970)
    epoch_start = datetime(1970, 1, 1, tzinfo=pytz.utc)
    
    # Convert microseconds to a timedelta
    time_since_epoch = timedelta(microseconds=int(timestamp_microseconds))
    
    # Calculate the datetime by adding the timedelta to the epoch start
    event_datetime_utc = epoch_start + time_since_epoch
    
    # Convert the UTC datetime to the specified timezone
    timezone_obj = pytz.timezone(timezone)
    event_datetime_tz = event_datetime_utc.astimezone(timezone_obj)
    
    # Format the datetime object into a string (hh:mm:ss, microseconds)
    formatted_string = event_datetime_tz.strftime("%H:%M:%S.%f")
    
    return formatted_string


def timestamp_to_formatted_string(timestamp_microseconds):
    # Assuming the timestamp is based on the UNIX epoch (January 1, 1970)
    epoch_start = datetime(1970, 1, 1)
    
    # Convert microseconds to a timedelta
    time_since_epoch = timedelta(microseconds=int(timestamp_microseconds))
    
    # Calculate the datetime by adding the timedelta to the epoch start
    event_datetime = epoch_start + time_since_epoch
    
    # Format the datetime object into a string (hh:mm:ss, microseconds)
    formatted_string = event_datetime.strftime("%H:%M:%S.%f")
    
    return formatted_string



class EventSensorIniVation:
    def __init__(self):
        self.address = '127.0.0.1'
        self.port = int(os.getenv('DV_PORT', '7777'))
        self.port_f = int(os.getenv('DV_PORT_FRAME', '7778'))
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


    def record_events_with_auxiliary_data(self, output_file_name, get_current_status, recording_duration_sec=None, timezone='Asia/Hong_Kong'):
        # Ensure the 'data' directory exists
        data_directory = 'data'
        if not os.path.exists(data_directory):
            os.makedirs(data_directory, exist_ok=True)

        # List to hold event data
        events_data = []

        with NetworkEventInput(address=self.address, port=self.port) as event_input:
            start_time = time.time()
            for event in event_input:
                if not self.recording_enabled:
                    break  # Stop recording if the flag is set to False

                # Capture the current system time in microseconds and convert to formatted string
                system_timestamp_microseconds = int(time.time() * 1e6)
                system_timestamp_formatted = timestamp_to_formatted_string_with_timezone(system_timestamp_microseconds, timezone)
                event_timestamp_formatted = timestamp_to_formatted_string_with_timezone(event.timestamp, timezone)

                # Append the formatted timestamps and event data to the events_data list
                events_data.append([event_timestamp_formatted, system_timestamp_formatted, event.x, event.y, event.polarity])
                
                if recording_duration_sec is not None and time.time() - start_time > recording_duration_sec:
                    break

        # Path to save the output file
        output_file_path = os.path.join(data_directory, f"{output_file_name}.csv")

        # Save the collected data to disk
        with open(output_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["event_timestamp", "system_timestamp", "x", "y", "polarity"])
            writer.writerows(events_data)

        print(f"Events recorded to {output_file_path}")

    # def record_events_with_auxiliary_data(self, output_file_name, recording_duration_sec=None, timezone='Asia/Hong_Kong'):
    #     data_directory = 'data'
    #     if not os.path.exists(data_directory):
    #         os.makedirs(data_directory, exist_ok=True)
    #     events_data = []
    #     output_file_path = os.path.join(data_directory, f"{output_file_name}.csv")
    #     with NetworkEventInput(address=self.address, port=self.port) as event_input:
    #         start_time = time.time()
    #         for event in event_input:
    #             if not self.recording_enabled:
    #                 break
    #             system_timestamp_microseconds = int(time.time() * 1e6)
    #             system_timestamp_formatted = timestamp_to_formatted_string_with_timezone(system_timestamp_microseconds, timezone)
    #             event_timestamp_formatted = timestamp_to_formatted_string_with_timezone(event.timestamp, timezone)
    #             events_data.append([event_timestamp_formatted, system_timestamp_formatted, event.x, event.y, event.polarity])
    #             if recording_duration_sec is not None and time.time() - start_time > recording_duration_sec:
    #                 break
    #     with open(output_file_path, 'w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["event_timestamp", "system_timestamp", "x", "y", "polarity"])
    #         writer.writerows(events_data)

    # def record_frames_to_numpy(self, output_file_name, recording_duration_sec=None):
    #     data_directory = 'data'
    #     if not os.path.exists(data_directory):
    #         os.makedirs(data_directory, exist_ok=True)
    #     frames_data = []
    #     with NetworkFrameInput(address=self.address, port=self.port_f) as frame_input:
    #         start_time = time.time()
    #         for frame in frame_input:
    #             if not self.recording_enabled:
    #                 break
    #             frames_data.append(frame.image)
    #             if recording_duration_sec is not None and time.time() - start_time > recording_duration_sec:
    #                 break
    #     np.save(os.path.join(data_directory, f"{output_file_name}.npy"), frames_data)

    def record_frames_with_auxiliary_data(self, output_file_name, recording_duration_sec=None, timezone='Asia/Hong_Kong'):
        data_directory = 'data'
        if not os.path.exists(data_directory):
            os.makedirs(data_directory, exist_ok=True)
        
        frames_data = []
        timestamps_data = []

        with NetworkFrameInput(address=self.address, port=self.port_f) as frame_input:
            start_time = time.time()
            for frame in frame_input:
                if not self.recording_enabled:
                    break

                # Capture the current system time in microseconds and convert to formatted string
                system_timestamp_microseconds = int(time.time() * 1e6)
                system_timestamp_formatted = timestamp_to_formatted_string_with_timezone(system_timestamp_microseconds, timezone)
                frame_timestamp_formatted = timestamp_to_formatted_string_with_timezone(frame.timestamp, timezone)

                # Store frame as an array (flattened if necessary) and timestamp data
                # frame_array = frame.image.flatten()  # Flatten the frame image array if necessary
                # frames_data.append(frame_array)
                frames_data.append(frame.image)
                timestamps_data.append([frame_timestamp_formatted, system_timestamp_formatted])

                if recording_duration_sec is not None and time.time() - start_time > recording_duration_sec:
                    break
        
        # Save frame data as a numpy array file
        np.save(os.path.join(data_directory, f"{output_file_name}.npy"), frames_data)
        
        # Save timestamps data as a CSV file
        timestamps_file_path = os.path.join(data_directory, f"{output_file_name}_timestamps.csv")
        with open(timestamps_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frame_timestamp", "system_timestamp"])
            writer.writerows(timestamps_data)

        print(f"Frames and timestamps recorded to {data_directory}")

    def record_events_and_frames_concurrently(self, event_file_name, frame_file_name, recording_duration_sec=None):
        event_thread = threading.Thread(target=self.record_events_with_auxiliary_data, args=(event_file_name, recording_duration_sec))
        # frame_thread = threading.Thread(target=self.record_frames_to_numpy, args=(frame_file_name, recording_duration_sec))
        frame_thread = threading.Thread(target=self.record_frames_with_auxiliary_data, args=(frame_file_name, recording_duration_sec))
        event_thread.start()
        frame_thread.start()
        event_thread.join()
        frame_thread.join()



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



if __name__ == '__main__':
    
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
