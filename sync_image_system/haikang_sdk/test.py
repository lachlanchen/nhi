#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_and_record.py

Enumerates MV-CB120-10UC-B camera, captures live images with timestamp overlay,
records to AVI, and optionally saves individual PNGs.
"""

import os, sys

# point at the SDK folder so we can import MvImport.*
sdk_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Python')
if sdk_root not in sys.path:
    sys.path.insert(0, sdk_root)


import sys
import cv2
import numpy as np
from ctypes import *
from datetime import datetime
from MvImport.MvCameraControl_class import (
    MvCamera, MV_CC_DEVICE_INFO_LIST, MV_GIGE_DEVICE, MV_USB_DEVICE,
    MVCC_INTVALUE, MVCC_ENUMVALUE, MVCC_FLOATVALUE, MV_CC_RECORD_PARAM,
    MV_FormatType_AVI, MV_TRIGGER_MODE_OFF, MV_CC_INPUT_FRAME_INFO,
    MvGvspPixelType
)

def error_and_exit(msg, code=1):
    print(msg)
    sys.exit(code)

def find_and_open_camera(target_model="MV-CB120-10UC-B"):
    # Initialize SDK
    MvCamera.MV_CC_Initialize()

    # Enumerate devices :contentReference[oaicite:0]{index=0}
    dev_list = MV_CC_DEVICE_INFO_LIST()
    tlayer = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayer, dev_list)
    if ret != 0:
        error_and_exit(f"EnumDevices failed: 0x{ret:02x}")
    if dev_list.nDeviceNum == 0:
        error_and_exit("No camera found")

    # Look for model name
    for i in range(dev_list.nDeviceNum):
        info = cast(dev_list.pDeviceInfo[i], POINTER(MVCC_DEVICE_INFO)).contents
        name_bytes = info.SpecialInfo.stUsb3VInfo.chModelName \
                     if info.nTLayerType & MV_USB_DEVICE else info.SpecialInfo.stGigEInfo.chModelName
        model = bytes(name_bytes).split(b'\x00',1)[0].decode()
        if model == target_model:
            cam = MvCamera()
            ret = cam.MV_CC_CreateHandle(info)
            if ret != 0:
                error_and_exit(f"CreateHandle failed: 0x{ret:02x}")
            ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                error_and_exit(f"OpenDevice failed: 0x{ret:02x}")
            # Turn off trigger
            cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            return cam
    error_and_exit(f"Camera model '{target_model}' not found")

def configure_and_start_record(cam, filename="output.avi"):
    # Get width & height
    st_int = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("Width", st_int)
    w = st_int.nCurValue
    cam.MV_CC_GetIntValue("Height", st_int)
    h = st_int.nCurValue

    # Get pixel format
    st_enum = MVCC_ENUMVALUE()
    cam.MV_CC_GetEnumValue("PixelFormat", st_enum)
    pix = MvGvspPixelType(st_enum.nCurValue)

    # Get resulting frame rate
    st_float = MVCC_FLOATVALUE()
    cam.MV_CC_GetFloatValue("ResultingFrameRate", st_float)
    fps = st_float.fCurValue

    # Set up recording parameters and start :contentReference[oaicite:1]{index=1}
    rec = MV_CC_RECORD_PARAM()
    memmove(byref(rec), b'\x00'*sizeof(rec), sizeof(rec))
    rec.nWidth      = w
    rec.nHeight     = h
    rec.enPixelType = pix
    rec.fFrameRate  = fps
    rec.nBitRate    = 4000000
    rec.enRecordFmtType = MV_FormatType_AVI
    rec.strFilePath= filename.encode('ascii')
    ret = cam.MV_CC_StartRecord(rec)
    if ret != 0:
        error_and_exit(f"StartRecord failed: 0x{ret:02x}")

    # Start grabbing
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        error_and_exit(f"StartGrabbing failed: 0x{ret:02x}")

    # Also prepare OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    return writer, w, h

def main():
    cam = find_and_open_camera()
    writer, w, h = configure_and_start_record(cam, "recording_timestamped.avi")

    # Buffer for BGR frames
    buf_size = w * h * 3
    buf = (c_ubyte * buf_size)()
    out_info = MV_CC_INPUT_FRAME_INFO()

    print("Press Ctrl+C to stop")
    try:
        while True:
            # Get one frame of BGR data :contentReference[oaicite:2]{index=2}
            st_frame = MV_FRAME_OUT()
            ret = cam.MV_CC_GetImageBuffer(st_frame, 1000)
            if ret != 0 or not st_frame.pBufAddr:
                continue

            ret = cam.MV_CC_GetImageForBGR(
                buf, buf_size, st_frame, 1000
            )
            cam.MV_CC_FreeImageBuffer(st_frame)
            if ret != 0:
                continue

            # Wrap into numpy
            arr = np.frombuffer(buf, dtype=np.uint8)
            frame = arr.reshape((h, w, 3))

            # Overlay timestamp
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(frame, ts, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Write to video
            writer.write(frame)

            # (Optional) display live
            cv2.imshow("Live", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping...")

    # Cleanup
    writer.release()
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_StopRecord()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
