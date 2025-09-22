#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
from ctypes import *

# Make sure your Python/MvImport folder is on the path:
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, "Python"))
sys.path.append(os.path.join(BASE, "Python", "MvImport"))

from MvCameraControl_class import (
    MvCamera,
    MV_CC_DEVICE_INFO_LIST,
    MV_CC_DEVICE_INFO,
    MV_GIGE_DEVICE,
    MV_USB_DEVICE,
    MV_GENTL_CAMERALINK_DEVICE,
    MV_GENTL_CXP_DEVICE,
    MV_GENTL_XOF_DEVICE,
    MV_FRAME_OUT,
    MV_CC_RECORD_PARAM,
    MV_CC_INPUT_FRAME_INFO,
    MVCC_INTVALUE,
    MVCC_ENUMVALUE,
    MVCC_FLOATVALUE,
    MvGvspPixelType
)
from CameraParams_const import MV_ACCESS_Exclusive
from CameraParams_header import MV_TRIGGER_MODE_OFF, MV_FormatType_AVI

# Global flag for recording thread
g_bExit = False

def recording_thread(cam, record_duration=10):
    """Thread function to capture frames and record video"""
    global g_bExit
    
    frame_out = MV_FRAME_OUT()
    memset(byref(frame_out), 0, sizeof(frame_out))
    
    input_frame_info = MV_CC_INPUT_FRAME_INFO()
    memset(byref(input_frame_info), 0, sizeof(MV_CC_INPUT_FRAME_INFO))
    
    start_time = time.time()
    frame_count = 0
    
    while not g_bExit and (time.time() - start_time) < record_duration:
        ret = cam.MV_CC_GetImageBuffer(frame_out, 1000)
        if frame_out.pBufAddr and ret == 0:
            frame_count += 1
            print(f"Recording frame {frame_count}: {frame_out.stFrameInfo.nWidth}×{frame_out.stFrameInfo.nHeight}")
            
            # Input frame data to recording
            input_frame_info.pData = cast(frame_out.pBufAddr, POINTER(c_ubyte))
            input_frame_info.nDataLen = frame_out.stFrameInfo.nFrameLen
            
            ret = cam.MV_CC_InputOneFrame(input_frame_info)
            if ret != 0:
                print(f"Input frame failed! ret=0x{ret:08x}")
            
            cam.MV_CC_FreeImageBuffer(frame_out)
        else:
            print(f"No data [0x{ret:08x}]")

def main():
    global g_bExit
    
    # 1) Initialize SDK
    MvCamera.MV_CC_Initialize()

    # 2) Enumerate devices
    dev_list = MV_CC_DEVICE_INFO_LIST()
    layers = (
        MV_GIGE_DEVICE
        | MV_USB_DEVICE
        | MV_GENTL_CAMERALINK_DEVICE
        | MV_GENTL_CXP_DEVICE
        | MV_GENTL_XOF_DEVICE
    )
    ret = MvCamera.MV_CC_EnumDevices(layers, dev_list)
    if ret != 0 or dev_list.nDeviceNum == 0:
        print(f"No devices found (ret=0x{ret:08x}).")
        MvCamera.MV_CC_Finalize()
        return

    print(f"Found {dev_list.nDeviceNum} device(s):")
    for i in range(dev_list.nDeviceNum):
        info = cast(dev_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        raw = (
            info.SpecialInfo.stGigEInfo.chModelName
            if info.nTLayerType == MV_GIGE_DEVICE
            else info.SpecialInfo.stUsb3VInfo.chModelName
        )
        name = bytes(raw).split(b'\x00',1)[0].decode(errors='ignore')
        print(f"  [{i}] {name}")

    idx = int(input(f"Select device [0–{dev_list.nDeviceNum-1}]: "))
    if not (0 <= idx < dev_list.nDeviceNum):
        print("Invalid index.")
        MvCamera.MV_CC_Finalize()
        return

    # 3) Create handle & open device
    info = cast(dev_list.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
    cam = MvCamera()
    if cam.MV_CC_CreateHandle(info) != 0 or cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
        print("Failed to open device.")
        MvCamera.MV_CC_Finalize()
        return

    # 4) For GigE, set optimal packet size
    if info.nTLayerType == MV_GIGE_DEVICE:
        pkt = cam.MV_CC_GetOptimalPacketSize()
        if pkt > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt)
        else:
            print(f"Warning: could not get packet size (pkt={pkt})")

    # 5) Turn off trigger
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    # 6) Get camera parameters for recording
    record_param = MV_CC_RECORD_PARAM()
    memset(byref(record_param), 0, sizeof(MV_CC_RECORD_PARAM))
    
    # Get width
    st_param = MVCC_INTVALUE()
    memset(byref(st_param), 0, sizeof(MVCC_INTVALUE))
    ret = cam.MV_CC_GetIntValue("Width", st_param)
    if ret != 0:
        print(f"Get width failed! ret=0x{ret:08x}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        return
    record_param.nWidth = st_param.nCurValue
    
    # Get height
    ret = cam.MV_CC_GetIntValue("Height", st_param)
    if ret != 0:
        print(f"Get height failed! ret=0x{ret:08x}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        return
    record_param.nHeight = st_param.nCurValue
    
    # Get pixel format
    st_enum_value = MVCC_ENUMVALUE()
    memset(byref(st_enum_value), 0, sizeof(MVCC_ENUMVALUE))
    ret = cam.MV_CC_GetEnumValue("PixelFormat", st_enum_value)
    if ret != 0:
        print(f"Get PixelFormat failed! ret=0x{ret:08x}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        return
    record_param.enPixelType = MvGvspPixelType(st_enum_value.nCurValue)
    
    # Get frame rate
    st_float_value = MVCC_FLOATVALUE()
    memset(byref(st_float_value), 0, sizeof(MVCC_FLOATVALUE))
    ret = cam.MV_CC_GetFloatValue("ResultingFrameRate", st_float_value)
    if ret != 0:
        print(f"Get ResultingFrameRate failed! ret=0x{ret:08x}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        return
    record_param.fFrameRate = st_float_value.fCurValue
    
    # Set recording parameters
    record_param.nBitRate = 5000  # Bitrate in kbps (you can adjust this)
    record_param.enRecordFmtType = MV_FormatType_AVI
    
    # Create filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.avi"
    record_param.strFilePath = filename.encode('ascii')
    
    print(f"\nRecording parameters:")
    print(f"  Resolution: {record_param.nWidth}×{record_param.nHeight}")
    print(f"  Frame rate: {record_param.fFrameRate:.2f} fps")
    print(f"  Bit rate: {record_param.nBitRate} kbps")
    print(f"  File: {filename}")
    
    # Get recording duration from user
    duration = int(input("\nEnter recording duration in seconds (default 10): ") or "10")
    
    # 7) Start recording
    ret = cam.MV_CC_StartRecord(record_param)
    if ret != 0:
        print(f"Start recording failed! ret=0x{ret:08x}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        return
    
    # 8) Start grabbing
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("StartGrabbing failed.")
        cam.MV_CC_StopRecord()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
        return
    
    # 9) Start recording thread
    print(f"\nRecording for {duration} seconds...")
    g_bExit = False
    
    try:
        thread_handle = threading.Thread(target=recording_thread, args=(cam, duration))
        thread_handle.start()
        thread_handle.join()
    except Exception as e:
        print(f"Error in recording thread: {e}")
    
    # 10) Stop everything
    g_bExit = True
    
    print("\nStopping recording...")
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_StopRecord()
    
    print(f"Video saved as: {filename}")
    
    # 11) Clean up
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()

if __name__ == "__main__":
    main()