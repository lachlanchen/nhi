#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
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
    MV_SAVE_IMAGE_TO_FILE_PARAM_EX,
    MV_Image_Bmp
)
from CameraParams_const import MV_ACCESS_Exclusive
from CameraParams_header import MV_TRIGGER_MODE_OFF

def main():
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

    # 4) For GigE, set optimal packet size to avoid timeouts
    if info.nTLayerType == MV_GIGE_DEVICE:
        pkt = cam.MV_CC_GetOptimalPacketSize()
        if pkt > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt)
        else:
            print(f"Warning: could not get packet size (pkt={pkt})")

    # 5) Turn off trigger & start grabbing
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if cam.MV_CC_StartGrabbing() != 0:
        print("StartGrabbing failed.")
    else:
        # 6) Grab one frame
        frame = MV_FRAME_OUT()
        memset(byref(frame), 0, sizeof(frame))
        if cam.MV_CC_GetImageBuffer(frame, 2000) != 0 or not frame.pBufAddr:
            print("GetImageBuffer failed.")
        else:
            w = frame.stFrameInfo.nWidth
            h = frame.stFrameInfo.nHeight
            fn = frame.stFrameInfo.nFrameNum
            print(f"Captured frame: {w}×{h}, #{fn}")

            # 7) Prompt and save as BMP
            if input("Save frame? (y/n): ").strip().lower() == "y":
                sp = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
                memset(byref(sp), 0, sizeof(sp))
                sp.nWidth      = w
                sp.nHeight     = h
                sp.enPixelType = frame.stFrameInfo.enPixelType
                sp.pData       = frame.pBufAddr
                sp.nDataLen    = frame.stFrameInfo.nFrameLen
                sp.enImageType = MV_Image_Bmp
                sp.pcImagePath = create_string_buffer(b"frame.bmp")
                sp.iMethodValue = 1

                ret2 = cam.MV_CC_SaveImageToFileEx(sp)
                if ret2 != 0:
                    print(f"SaveImage failed (ret=0x{ret2:08x}).")
                else:
                    print("Saved frame.bmp")

        cam.MV_CC_FreeImageBuffer(frame)
        cam.MV_CC_StopGrabbing()

    # 8) Clean up
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()

if __name__ == "__main__":
    main()
