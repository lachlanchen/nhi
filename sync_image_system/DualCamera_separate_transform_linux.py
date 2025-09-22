#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linux/Ubuntu variant of DualCamera_separate_transform.py

- Uses the Linux wrapper (MvCameraControl_class_linux) which loads
  libMvCameraControl.so via ctypes.CDLL.
- Leaves the original Windows code untouched.
"""

import os
import sys

# Ensure SDK Python paths are on sys.path
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, "haikang_sdk", "Python"))
sys.path.append(os.path.join(BASE, "haikang_sdk", "Python", "MvImport"))

# Import everything from the original file, except use Linux wrapper name
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
import platform
from datetime import datetime

try:
    from ctypes import *
    import numpy as np
    import cv2
    # Linux: import the CDLL-based wrapper
    from MvCameraControl_class_linux import (
        MvCamera, MV_CC_DEVICE_INFO_LIST, MV_CC_DEVICE_INFO,
        MV_GIGE_DEVICE, MV_USB_DEVICE, MV_GENTL_CAMERALINK_DEVICE,
        MV_GENTL_CXP_DEVICE, MV_GENTL_XOF_DEVICE, MV_FRAME_OUT,
        MV_CC_RECORD_PARAM, MV_CC_INPUT_FRAME_INFO, MV_CC_PIXEL_CONVERT_PARAM,
        MVCC_INTVALUE, MVCC_ENUMVALUE, MVCC_FLOATVALUE, MvGvspPixelType
    )
    from CameraParams_const import MV_ACCESS_Exclusive
    from CameraParams_header import MV_TRIGGER_MODE_OFF, MV_FormatType_AVI
    from PixelType_header import PixelType_Gvsp_Mono8, PixelType_Gvsp_RGB8_Packed
    FRAME_CAMERA_AVAILABLE = True
except ImportError as e:
    print(f"Frame camera SDK not available on Linux: {e}")
    FRAME_CAMERA_AVAILABLE = False

# Event camera modules (optional)
try:
    from metavision_core.event_io.raw_reader import initiate_device
    from metavision_core.event_io import EventsIterator
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
    EVENT_CAMERA_AVAILABLE = True
except ImportError as e:
    print(f"Event camera SDK not available: {e}")
    EVENT_CAMERA_AVAILABLE = False

# Platform handling for always-on-top (Windows only)
WINDOWS_AVAILABLE = False

# Bring over the rest of the implementation by importing and executing
# the shared implementation from the Windows script, but we need the
# class and function bodies. To avoid code duplication, we include the
# logic from the original file by reading and exec-ing it with the
# import line swapped. However, for safety and simplicity, we just
# copy a minimal shim that imports and runs the original module's main
# if needed. The Linux wrapper above provides the same API.

# Reuse the same implementation by importing the original module as a
# library. We ensure it sees the Linux wrapper by shadowing the name.

# Map name in sys.modules so that imports inside the original file
# resolve to our Linux wrapper where applicable.
import types
sys.modules['MvCameraControl_class'] = sys.modules.get('MvCameraControl_class_linux')

# Now import the original implementation module and expose its globals
from importlib import import_module
_impl = import_module('DualCamera_separate_transform')

# Re-export attributes so running this file behaves like the original
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('_')})

if __name__ == '__main__':
    if hasattr(_impl, 'main'):
        _impl.main()
