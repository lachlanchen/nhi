# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 18:48:12 2022

@author: eee
"""
import cv2
import numpy as np
import glob,os
import pandas as pd
from dv import AedatFile

reprocess = 1
frame_out = 1
# data_path = '../data/24-20220609(Autofocusing-glass_slide_with_marker)/event/pulsed_laser/'
data_path = 'E:\\Expe_data\\0110\\dvs\\'
event_file = glob.glob(data_path+'*.aedat4')[-2]
# event_file = 'D:\WHK\e0105\e1\dvSave-2023_01_05_12_24_48.aedat4'

if os.path.isfile(data_path+'davis.csv') and  not reprocess:
    event_series = pd.read_csv(glob.glob(data_path+'davis.csv')[-1], sep=",").values
else:
    with AedatFile(event_file) as f:
        events = np.hstack([packet for packet in f['events'].numpy()])
        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
    out_event = np.vstack([timestamps, x, y, polarities]).T
    event_series = out_event[:, [1,2,3,0]]
    # event_series[:,3] -= int(int(event_series[0,3]/1e8)*1e8)
    event_series[:,3] -= event_series[0,3]
    df = pd.DataFrame(event_series)
    df.to_csv(data_path+'davis.csv',index=0, header=0)
    
frame_dir = data_path + 'frames\\'
if (not os.path.isdir(frame_dir)) & frame_out:
    os.mkdir(frame_dir)
    frame_list = []
    with AedatFile(event_file) as f:
        for frame in f['frames']:
            frame_list.append(frame)
    f_start = frame_list[0].timestamp
    for frame in frame_list:
        file_name = data_path + 'frames\\' + str(frame.timestamp - f_start).zfill(8) + '.bmp'
        cv2.imwrite(file_name, frame.image)
                