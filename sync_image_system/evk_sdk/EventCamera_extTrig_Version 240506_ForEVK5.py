#!/usr/bin/env python
# coding: utf-8

# import metavision_designer_engine as mvd_engine
# import metavision_designer_core as mvd_core
import metavision_core as mvd_core
import metavision_hal as mv_hal
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io.raw_reader import RawReader
# import metavision_hal.I_TriggerIn as trigger_in
# from metavision_core.event_io.raw_reader import initiate_device
# from metavision_core.event_io.raw_reader import RawReader


import numpy as np
import time
# from metavision_designer_core import FileProducerTrigger as filereader
# from metavision_core import FileProducerTrigger as filereader


class EventCamera:
    def __init__(self,input_path,extTrig=0,delta_t=1e3):
        '''
            Initialize the camera
        '''

        # file path being read, it will open a real camera when input_path=''
        self.input_path=input_path
        
        # int, channel of external trigger signal; by default extTrig=0
        self.extTrig=extTrig
#         self.delta_t=delta_t
        
        # open a camera with initialization, returning device ID
        self.device = initiate_device(path=self.input_path)
        i_trigger_in = self.device.get_i_trigger_in()
        # i_trigger_in.enable(0)
        # i_trigger_in.enable(metavision_hal.I_TriggerIn.Channel.MAIN)
        print(i_trigger_in.enable(mv_hal.I_TriggerIn.Channel.MAIN))
        print(i_trigger_in.is_enabled(mv_hal.I_TriggerIn.Channel.MAIN))
#       self.mv_iterator = EventsIterator.from_device(device=self.device,delta_t=self.delta_t)
   
        # instantiation a reader for reading data from camera buffer
        self.my_reader= RawReader.from_device(device=self.device,max_events=1000000000)
   
        # instantiation a bias_setting, for setting threshold
        self.bias=self.device.get_i_ll_biases()
        # self.bias.set('bias_diff_off', 225)
        # self.bias.set('bias_diff_on', 375)
   
        # instantiation a trigger control to enable/disable external trigger.
        self.trigger = self.device.get_i_trigger_in()
   
        # enable the channel( default 6) to recive trigger signals from trigger_out
        # self.trigger.enable(0)
   
        # initialize a event reader 
        self.i_events_stream = self.device.get_i_events_stream()
   
        # instantiation a controller to start/stop producing event from the camera side.
        # self.control=self.device.get_i_device_control()
    
    def log_data(self,file_raw,flag_time):
        self.my_reader.clear_ext_trigger_events()
        self.i_events_stream.log_raw_data(file_raw) # save events into file named by file_raw
        self.events_iterator = EventsIterator.from_device(self.device)
        self.height, self.width = self.events_iterator.get_size()
        print('size=',self.height,self.width)
        flag=0
        for evs in self.events_iterator:
            print(np.max(evs['p'])) # print out event polarity for monitoring the working status
            flag=flag+1
            if flag >flag_time: # looping from 0 to flag_time to control the total sweeping time
                break

    def setThreshold(self,bias_off,bias_on,bias_fo,bias_diff):
        '''
        changing threshold; 
        bias_off: threshold of decrement intensity, range: 0-300
        bias_on: threshold of increment intensity, range: 300-600
        '''
        # self.bias.set('bias_diff_off',bias_off)
        # self.bias.set('bias_diff_on',bias_on)
        # self.bias.set('bias_fo', bias_fo)
        # self.bias.set('bias_diff', bias_diff)
        print('bias_value=',self.bias.get_all_biases())
        
    def setTrigger(self):
        self.trigger.enable(self.extTrig)
        
    # def StartorStop(self,start):
    #     '''
    #     start/stop producing event from the camera side
    #     every time to start, the external trigger should be re-enabled
    #     '''
    #     if start:
    #         # self.my_reader.clear_ext_trigger_events()
    #         self.trigger.enable(self.extTrig)
    #         self.control.start()
    #     else:
    #         self.control.stop()
            
    # def CollectandSave(self,file,file_event,delta_t):
    #     '''
    #     collecting data from the camera buffer
    #     the starting point is from the trigger event
    #     while duration is the time used for scanning
    #     '''
    #     delta_t=delta_t*1e6
    #     self.TrigEvent=self.my_reader.get_ext_trigger_events()
    #     # self.my_reader.current_time=self.TrigEvent['t'][0]
    #     print(self.TrigEvent)
    #     self.RawEvent=self.my_reader.load_delta_t(delta_t)
    #     print(np.where(self.RawEvent['p']>1))
    #     np.save(file,self.RawEvent)
    #     np.save(file_event,self.TrigEvent)

            
if __name__ == "__main__":
    
    # create a new camera: 
    # it will open a real camera when input_path=''
    # otherwise will open a recorded file
    # input_path='C:\\Users\\13925\\2021-07-06 19-33-38.raw'
    input_path = ''
    my_camera = EventCamera(input_path)


    # set threshold:
    # theshold is defined as: for positive event: threshold= thr_on-300; for negative events: threshold= 300-thr_off
    # bwtter to keep the default value: thr_on =375; thr_off=225
    # threshold=75
    # thr_on=50
    # thr_off=50
    # bias_diff_new=0
    # threshold=(thr_on-thr_off)/2  # later will be transferre as string for naming the recording file
    # # set the cut-off frequency, the lower, the faster response; but better to keep the default value: 1725
    # bias_fo_new=1725 
    # # set threhold
    # my_camera.setThreshold(thr_off,thr_on,bias_fo_new,bias_diff_new)
    
    
    ############### soem parameters that may need to be changed every time ################
    laser='1' # laser power (mW) that will be shown in the name of recorded file
    period='75' # sweeping period time (ms, single direction) that will be shown in the name of recorded file
    direction=np.array(['forward','backward'])
    sweep_dir=direction[0] # sweeping direction (0: forward; 1: backward) that will be shown in the name of recorded file
    flag_time=50 # set the total sweeping time, usually flag_time=300 means 225 loops of sweep with period of 70ms
    # path and name of the file to record the events:
    # file = 'C:\\Users\\13925\\Desktop\\Data\\Event_camera_220628\\test\\' + '%smW_%sms_%s_thr%s_fo%s_AWG_2835-2905_red10mW-2.raw' % (laser,period,sweep_dir,threshold,bias_fo_new)
    # file = 'C:\\madhav\\Event Camera\\09092023\\' +'%s%sV%sms_%sthreshold_%sflag.raw' % (sweep_dir,laser,period,str(threshold),str(flag_time))
    # file = 'D:\\Y2024_ExperimentData\\08-EVK5_Test\\Python_Program_test\\ODMR\\05102024\\' + 'NewScheme(500us)_%sV%sms_%sthreshold_%sflag_test6.raw' % (laser, period, str(threshold),str(flag_time))
    file = 'D:\\Y2024_ExperimentData\\08-EVK5_Test\\ODMR\\05102024\\trial.raw'

    # # used ot be used to store trigger events, but now is discarded; 
    # file_event='C:\\Users\\13925\\'+'%s_event.npy'% (time.strftime("%Y-%m-%d %H-%M-%S",time.localtime()))
    
    # start capaturing events
    my_camera.log_data(file,flag_time)
    
