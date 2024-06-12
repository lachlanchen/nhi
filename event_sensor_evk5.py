#!/usr/bin/env python
# coding: utf-8

import os
import csv
import argparse
from datetime import datetime, timedelta
import pytz
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import RawReader, initiate_device
import metavision_hal as mv_hal
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Process events from EVK5 sensor.")
    parser.add_argument('-i', '--input', type=str, default='', help='Input path for the event data source.')
    parser.add_argument('-d', '--duration', type=int, default=None, help='Recording duration in seconds, if applicable.')
    parser.add_argument('-z', '--timezone', type=str, default='Asia/Hong_Kong', help='Timezone for timestamp conversion.')
    parser.add_argument('-o', '--output', type=str, default='event_output.csv', help='Output CSV file name.')
    return parser.parse_args()

def timestamp_to_formatted_string_with_timezone(timestamp_microseconds, timezone='Asia/Hong_Kong'):
    epoch_start = datetime(1970, 1, 1, tzinfo=pytz.utc)
    time_since_epoch = timedelta(microseconds=int(timestamp_microseconds))
    event_datetime_utc = epoch_start + time_since_epoch
    timezone_obj = pytz.timezone(timezone)
    event_datetime_tz = event_datetime_utc.astimezone(timezone_obj)
    return event_datetime_tz.strftime("%H:%M:%S.%f")

class EventSensorEVK5:
    def __init__(self, input_path=''):
        self.input_path = input_path
        self.recording_enabled = False
        self.current_event_timestamp = None

        # Setup camera or device from the given input path
        self.device = initiate_device(path=self.input_path)
        self.reader = RawReader.from_device(self.device, max_events=1000000000)
        self.bias = self.device.get_i_ll_biases()
        self.trigger = self.device.get_i_trigger_in()


        # Enable the main trigger channel
        self.trigger.enable(mv_hal.I_TriggerIn.Channel.MAIN)
        print("Trigger enabled:", self.trigger.is_enabled(mv_hal.I_TriggerIn.Channel.MAIN))

        # self.setThreshold(100, 100)

    def setThreshold(self, bias_off, bias_on):
        """
        Set bias values for the device.
        Args:
            bias_off (int): Threshold for decrement intensity.
            bias_on (int): Threshold for increment intensity.
        """
        print('Bias values after setting:', self.bias.get_all_biases())
        self.bias.set('bias_diff_off', bias_off)
        self.bias.set('bias_diff_on', bias_on)
        print('Bias values after setting:', self.bias.get_all_biases())

    def start_recording(self):
        self.recording_enabled = True

    def stop_recording(self):
        self.recording_enabled = False

    def record_events_with_auxiliary_data(self, output_file_name, recording_duration_sec=None, timezone='Asia/Hong_Kong'):
        events_data = []
        data_directory = 'data'
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        output_file_path = os.path.join(data_directory, f"{output_file_name}.csv")
        start_time = time.time() 
        start_timestamp = int(start_time * 1e6)

        self.reader.clear_ext_trigger_events()
        self.events_iterator = EventsIterator.from_device(self.device)
        for events in self.events_iterator:
            if not self.recording_enabled:
                break

            system_timestamp_microseconds = int(time.time() * 1e6)
            system_timestamp_formatted = timestamp_to_formatted_string_with_timezone(system_timestamp_microseconds, timezone)
            
            for event in events:
                x, y, polarity, timestamp = event
                self.current_event_timestamp = timestamp + start_timestamp
                event_timestamp_formatted = timestamp_to_formatted_string_with_timezone(self.current_event_timestamp, timezone)
                events_data.append([event_timestamp_formatted, system_timestamp_formatted, x, y, polarity])

            if recording_duration_sec is not None and time.time() - start_time > recording_duration_sec:
                break

        with open(output_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["event_timestamp", "system_timestamp", "x", "y", "polarity"])
            writer.writerows(events_data)
        
        print(f"Events recorded to {output_file_path}")

    def record_events_and_frames_concurrently(self, event_file_name, frame_file_name, recording_duration_sec=None):
        """
        Records events and frames concurrently. Currently, only events are recorded as EVK5 does not output frames.
        """
        self.record_events_with_auxiliary_data(event_file_name, recording_duration_sec=recording_duration_sec)

    def start_event_processing(self, output_file_name, recording_duration_sec, timezone):
        self.start_recording()
        self.record_events_with_auxiliary_data(output_file_name, recording_duration_sec, timezone)
        self.stop_recording()

if __name__ == "__main__":
    args = parse_args()
    sensor = EventSensorEVK5(input_path=args.input)
    sensor.start_event_processing(args.output, args.duration, args.timezone)
