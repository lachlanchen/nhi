# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 19:37:13 2020

@author: eee
"""

import numpy as np
import time
import glob,os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def form_bii(event_sel, roi):
    bii = np.zeros(((roi[1]-roi[0]),(roi[3]-roi[2]),2),dtype=int)
    lines = event_sel.shape[0]
    for ind in range(0,lines):
        event_ind = event_sel[ind]
        if int(event_ind[3]) == 1: #positive
            if bii[event_ind[1]-roi[0],event_ind[0]-roi[2],0] == 0:
                bii[event_ind[1]-roi[0],event_ind[0]-roi[2],0] = 1
            else:
                bii[event_ind[1]-roi[0],event_ind[0]-roi[2],0] = bii[event_ind[1]-roi[0],event_ind[0]-roi[2],0]+1 # += 1
        elif int(event_ind[3]) == -1:           #negative
            if bii[event_ind[1]-roi[0],event_sel[ind,0]-roi[2],1] == 0:
                bii[event_ind[1]-roi[0],event_sel[ind,0]-roi[2],1] = 1
            else:
                bii[event_ind[1]-roi[0],event_sel[ind,0]-roi[2],1] = bii[event_ind[1]-roi[0],event_ind[0]-roi[2],1]+1 # += 1
    normalized_bii = bii - bii.mean()
    return normalized_bii

def form_bii2(event_sel, roi):
    bii = np.zeros(((roi[1]-roi[0]),(roi[3]-roi[2])),dtype=int)
    for ind in range(0, event_sel.shape[0]):
        event_ind = event_sel[ind]
        if int(event_ind[3]) == 1 or int(event_ind[3]) == -1: #positive or negative
            bii[event_ind[1]-roi[0],event_ind[0]-roi[2]] = bii[event_ind[1]-roi[0],event_ind[0]-roi[2]]+1 # += 1
    bii_min = bii.min(axis=(0, 1), keepdims=True)
    bii_max = bii.max(axis=(0, 1), keepdims=True)
    normalized_bii = (bii - bii_min)/(bii_max - bii_min)
    return normalized_bii

def crop_events(event_sel, start_time, intervel_time, time_window):
    event_1 = event_sel[(event_sel[:,4]>=start_time) & (event_sel[:,4]<(start_time+time_window))]
    event_2 = event_sel[(event_sel[:,4]>=(start_time+intervel_time)) & (event_sel[:,4]<(start_time+intervel_time+time_window))]
    if event_1.shape[0] > event_2.shape[0]:
        event_1 = event_1[np.random.choice(event_1.shape[0], event_2.shape[0], replace=False), :]
#        event_1 = event_1[0:event_2.shape[0],:]
    else:
        event_2 = event_2[np.random.choice(event_2.shape[0], event_1.shape[0], replace=False), :]
#        event_2 = event_2[0:event_1.shape[0],:]

    return event_1,event_2

def corr2D(A,B):
    m = A.shape[0]
    n = A.shape[1]
    corr=abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(A)*np.conj(np.fft.fft2(B)))))/(n*m)
    return corr

data_path = '../data/0512/'
file_name = '2.5mm'
event_file = glob.glob(data_path+file_name+'.csv')[-1]
data = pd.read_csv(event_file, sep=",").values 
time_length = (data[-1][4] - data[0][4])
print("\nevent file is %s\ntime length is %sms\n" % (os.path.basename(event_file), time_length/1000))
###########################################################
## y(0-799),x(0-1279),intensity(0,4095),p(1,-1),time(us)
###########################################################

start_time = 0.86 * 1e6
time_window = 0.03 * 1e6
interval_time = 0.2 * 1e6
data_sel = data#[(data[:,3]>=start_time) & (data[:,3]<(start_time + duration))]
rol_size = 512
roi = [000,rol_size,000,rol_size] #[x1,x2,y1,y2] 
data_crop = data_sel[(data_sel[:,1]>roi[0]) & (data_sel[:,1]<roi[1]) & (data_sel[:,0]>roi[2]) & (data_sel[:,0]<roi[3])]
#bii_1 = form_bii(data_sel,roi,start_time,time_window)
event_a, event_b = crop_events(data_crop, start_time, interval_time, time_window)
bii_a = form_bii(event_a,roi)
bii_b = form_bii(event_b,roi)
bii_1 = np.transpose(np.squeeze(bii_a[:,:,0]+bii_a[:,:,1]))
bii_2 = np.transpose(np.squeeze(bii_b[:,:,0]+bii_b[:,:,1]))


corr = corr2D(bii_1,bii_2)
corr[int(rol_size/2),int(rol_size/2)] = 0
maximum = np.max(np.max(corr))
indices = np.where(corr == maximum)
dispx= 0.5*bii_1.shape[0] - indices[0]
dispy= 0.5*bii_1.shape[1] - indices[1]
print("disp from correlation is %s, %s" % (dispx, dispy))
#####################################################################
#fig = plt.figure()
#plt.imshow(corr)


#
#fig = plt.figure()
#data_plot = data_sel
#pos_rid = data_plot[:,3] > 0;
#neg_rid = data_plot[:,3] <= 0;
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(data_plot[pos_rid,0], data_plot[pos_rid,3]/1000,data_plot[pos_rid,1],c='r',s=5)
#ax.scatter(data_plot[neg_rid,0], data_plot[neg_rid,3]/1000,data_plot[neg_rid,1],c='b',s=5)
#
#ax.set_xlabel('X axis')
#ax.set_ylabel('Time [ms]')
#ax.set_zlabel('Y axis')
#
#plt.show()    
