#!/usr/bin/env python
# coding: utf-8

import os
import csv
import numpy as np
import time
import argparse
from datetime import datetime, timedelta
import pytz

from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device, RawReader
import metavision_hal as mv_hal

class EventCamera:
    def __init__(self, input_path, extTrig=0, delta_t=1e3):
        """
        Initialize the camera or open a raw file.
        Args:
            input_path (str): Path to the input file or '' for a real camera.
            extTrig (int): Channel of external trigger signal (default: 0).
            delta_t (float): Time step for the event iterator in microseconds (default: 1000).
        """
        self.input_path = input_path
        self.extTrig = extTrig
        
        # Open a camera or file and set up device
        self.device = initiate_device(path=self.input_path)
        self.trigger = self.device.get_i_trigger_in()
        print(self.trigger.enable(mv_hal.I_TriggerIn.Channel.MAIN))
        print(self.trigger.is_enabled(mv_hal.I_TriggerIn.Channel.MAIN))
        
        # Setup the raw reader for event data acquisition
        self.my_reader = RawReader.from_device(device=self.device, max_events=1000000000)
        self.bias = self.device.get_i_ll_biases()
        
    def log_data(self, file_raw, flag_time):
        """
        Logs data to a file and prints the max polarity of events for a set number of iterations.
        Args:
            file_raw (str): Filename to save raw data.
            flag_time (int): Number of iterations to capture data.
        """
        self.my_reader.clear_ext_trigger_events()
        self.device.get_i_events_stream().log_raw_data(file_raw)
        self.events_iterator = EventsIterator.from_device(self.device)
        self.height, self.width = self.events_iterator.get_size()
        print('Size=', self.height, self.width)
        
        flag = 0
        for evs in self.events_iterator:
            print(evs)
            # print(np.max(evs['p']))
            flag += 1
            if flag > flag_time:
                break

    def setThreshold(self, bias_off, bias_on, bias_fo, bias_diff):
        """
        Set bias values for the device.
        Args:
            bias_off (int): Threshold for decrement intensity.
            bias_on (int): Threshold for increment intensity.
            bias_fo (int): Cut-off frequency.
            bias_diff (int): Differential bias.
        """
        print('Bias values before setting:', self.bias.get_all_biases())
        self.bias.set('bias_diff_off', bias_off)
        self.bias.set('bias_diff_on', bias_on)
        # Additional bias settings can be uncommented and adjusted here.
        print('Bias values after setting:', self.bias.get_all_biases())

    def setTrigger(self):
        """
        Enable external trigger.
        """
        self.trigger.enable(self.extTrig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control and log data from an Event Camera.")
    parser.add_argument('-i', '--input', type=str, default='', help='Input path for the event data source.')
    parser.add_argument('-o', '--output', type=str, default='event_output.csv', help='Output CSV file name.')
    args = parser.parse_args()

    # Setup the camera
    my_camera = EventCamera(input_path=args.input)
    
    # Filename and number of iterations to log data
    output_file = args.output
    flag_time = 50  # Modify this as needed
    
    # Log data and print results
    my_camera.log_data(output_file, flag_time)
