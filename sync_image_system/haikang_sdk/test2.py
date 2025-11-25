#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from ctypes import cast, POINTER, c_ubyte, memset, sizeof

# Adjust this path if your folder structure differs.
# It makes sure Python can import MvImport.MvCameraControl_class
base_dir = os.path.dirname(os.path.abspath(__file__))
sdk_python = os.path.join(base_dir, 'Python')
if sdk_python not in sys.path:
    sys.path.insert(0, sdk_python)

from MvImport.MvCameraControl_class import (
    MvCamera,
    MV_CC_DEVICE_INFO_LIST,
    MV_GIGE_DEVICE,
    MV_USB_DEVICE,
    MV_FRAME_OUT,
    MV_SAVE_IMAGE_TO_FILE_PARAM_EX,
    MV_Image_Jpeg,
    MV_TRIGGER_MODE_OFF,
    MV_OK
)

def main():
    # 1) Initialize SDK
    MvCamera.MV_CC_Initialize()

    # 2) Enumerate devices
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    if ret != 0 or device_list.nDeviceNum == 0:
        print("No devices found or enum failed (0x%X)." % ret)
        MvCamera.MV_CC_Finalize()
        return

    print(f"Found {device_list.nDeviceNum} device(s), opening the first one…")

    # 3) Create handle & open first device
    cam = MvCamera()
    dev_info = cast(device_list.pDeviceInfo[0], POINTER(device_list.pDeviceInfo._type_)).contents
    ret = cam.MV_CC_CreateHandle(dev_info)
    if ret != 0:
        print("CreateHandle failed (0x%X)." % ret)
        cleanup(cam)
        return

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("OpenDevice failed (0x%X)." % ret)
        cleanup(cam)
        return

    # 4) Set continuous (free-run) mode
    cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")  # or use MV_TRIGGER_MODE_OFF :contentReference[oaicite:0]{index=0}

    # 5) Prepare output folder
    out_dir = os.path.join(base_dir, 'images')
    os.makedirs(out_dir, exist_ok=True)

    # 6) Start grabbing
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("StartGrabbing failed (0x%X)." % ret)
        cleanup(cam)
        return

    print("Grabbing… press Ctrl+C to stop and exit.")

    frame_num = 0
    try:
        while True:
            out_frame = MV_FRAME_OUT()
            memset(byref(out_frame), 0, sizeof(out_frame))

            ret = cam.MV_CC_GetImageBuffer(out_frame, 1000)
            if ret == MV_OK and out_frame.pBufAddr:
                # Save JPEG
                save_param = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
                save_param.enPixelType = out_frame.stFrameInfo.enPixelType
                save_param.nWidth       = out_frame.stFrameInfo.nWidth
                save_param.nHeight      = out_frame.stFrameInfo.nHeight
                save_param.nDataLen     = out_frame.stFrameInfo.nFrameLen
                save_param.pData        = cast(out_frame.pBufAddr, POINTER(c_ubyte))
                save_param.enImageType  = MV_Image_Jpeg      # JPEG format :contentReference[oaicite:1]{index=1}
                save_param.nQuality     = 80
                name = f"frame_{frame_num:06d}.jpg"
                path = os.path.join(out_dir, name)
                save_param.pcImagePath = path.encode('ascii')
                save_param.iMethodValue= 1

                ret = cam.MV_CC_SaveImageToFileEx(save_param)
                if ret == MV_OK:
                    print(f"Saved {name}")
                else:
                    print(f"Failed to save frame {frame_num} (0x%X)" % ret)

                frame_num += 1
                cam.MV_CC_FreeImageBuffer(out_frame)
            else:
                # timeout or error
                # print("No frame (0x%X)" % ret)
                continue

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # 7) Cleanup
    cleanup(cam)


def cleanup(cam):
    """Stop, close, destroy, finalize."""
    try:
        cam.MV_CC_StopGrabbing()
    except:
        pass
    try:
        cam.MV_CC_CloseDevice()
    except:
        pass
    try:
        cam.MV_CC_DestroyHandle()
    except:
        pass
    MvCamera.MV_CC_Finalize()
    print("Cleaned up, exiting.")


if __name__ == "__main__":
    main()
