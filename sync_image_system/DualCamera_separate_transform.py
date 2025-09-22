#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
import os
import sys
import platform
from datetime import datetime

# Add the camera SDK paths (modify these paths according to your setup)
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, "haikang_sdk", "Python"))
sys.path.append(os.path.join(BASE, "haikang_sdk", "Python", "MvImport"))

# Import frame camera modules
try:
    from ctypes import *
    import numpy as np
    import cv2
    from MvCameraControl_class import (
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
    print(f"Frame camera SDK not available: {e}")
    FRAME_CAMERA_AVAILABLE = False

# Import event camera modules
try:
    from metavision_core.event_io.raw_reader import initiate_device
    from metavision_core.event_io import EventsIterator
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
    EVENT_CAMERA_AVAILABLE = True
except ImportError as e:
    print(f"Event camera SDK not available: {e}")
    EVENT_CAMERA_AVAILABLE = False

# Platform-specific imports for always on top functionality
if platform.system() == "Windows":
    try:
        import win32gui
        import win32con
        WINDOWS_AVAILABLE = True
    except ImportError:
        WINDOWS_AVAILABLE = False
        print("win32gui not available. Install pywin32 for full always-on-top support")
else:
    WINDOWS_AVAILABLE = False


def set_window_always_on_top(window_title, always_on_top=True):
    """Set a window to always on top (Windows only)"""
    if not WINDOWS_AVAILABLE:
        return False
    
    try:
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd:
            if always_on_top:
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            else:
                win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            return True
    except Exception as e:
        print(f"Error setting window always on top: {e}")
    return False


class FrameCameraController:
    """Modified version of CameraRecorder for GUI integration"""
    
    def __init__(self, status_callback=None, screen_info=None):
        self.cam = None
        self.device_info = None
        self.recording_thread = None
        self.preview_thread = None
        self.command_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=5)
        self.is_recording = False
        self.is_grabbing = False
        self.show_preview = False
        self.exit_flag = False
        self.current_filename = None
        self.record_params = None
        self.stats_lock = threading.Lock()
        self.status_callback = status_callback
        self.screen_info = screen_info or {}
        
        # Always on top state - DEFAULT TO TRUE
        self.preview_always_on_top = True
        
        # Frame transformation settings - DEFAULT VERTICAL FLIP ON
        self.vertical_flip = True
        self.horizontal_flip = False
        self.rotation_angle = 0  # 0, 90, 180, 270
        self.use_opencv_recording = False  # Use OpenCV when transformations are applied
        self.opencv_writer = None
        
        self.stats = {
            'frames_captured': 0,
            'frames_displayed': 0,
            'recording_start_time': None,
            'last_fps': 0.0
        }
        
        # For pixel format conversion
        self.convert_param = MV_CC_PIXEL_CONVERT_PARAM()
        self.rgb_buffer = None
        self.rgb_buffer_size = 0
    
    def find_cameras(self):
        """Find available frame cameras"""
        if not FRAME_CAMERA_AVAILABLE:
            return []
        
        try:
            # Initialize SDK
            MvCamera.MV_CC_Initialize()
        except:
            pass
        
        dev_list = MV_CC_DEVICE_INFO_LIST()
        layers = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | 
                 MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
        
        ret = MvCamera.MV_CC_EnumDevices(layers, dev_list)
        if ret == 0:
            cameras = []
            for i in range(dev_list.nDeviceNum):
                device_info = cast(dev_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                name = self._get_device_name(device_info)
                cameras.append((i, name, device_info))
            return cameras
        return []
    
    def _get_device_name(self, device_info):
        """Extract device name from device info"""
        if device_info.nTLayerType == MV_GIGE_DEVICE:
            raw = device_info.SpecialInfo.stGigEInfo.chModelName
        elif device_info.nTLayerType == MV_USB_DEVICE:
            raw = device_info.SpecialInfo.stUsb3VInfo.chModelName
        elif device_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
            raw = device_info.SpecialInfo.stCMLInfo.chModelName
        elif device_info.nTLayerType == MV_GENTL_CXP_DEVICE:
            raw = device_info.SpecialInfo.stCXPInfo.chModelName
        elif device_info.nTLayerType == MV_GENTL_XOF_DEVICE:
            raw = device_info.SpecialInfo.stXoFInfo.chModelName
        else:
            return "Unknown_Device"
        
        return bytes(raw).split(b'\x00', 1)[0].decode(errors='ignore')
    
    def connect_camera(self, device_info):
        """Connect to a specific camera"""
        if not FRAME_CAMERA_AVAILABLE:
            return False
            
        try:
            # Force disconnect any existing camera first
            self.force_disconnect()
            
            # Initialize SDK if not already done
            try:
                MvCamera.MV_CC_Initialize()
            except:
                pass
            
            self.cam = MvCamera()
            self.device_info = device_info
            
            # Create handle
            if self.cam.MV_CC_CreateHandle(device_info) != 0:
                return False
            
            # Open device
            if self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
                self.cam.MV_CC_DestroyHandle()
                return False
            
            # For GigE, set optimal packet size
            if device_info.nTLayerType == MV_GIGE_DEVICE:
                pkt = self.cam.MV_CC_GetOptimalPacketSize()
                if pkt > 0:
                    self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt)
            
            # Turn off trigger mode
            self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            
            # Setup recording parameters
            self._setup_recording_params()
            
            self._notify_status("Frame camera connected")
            return True
            
        except Exception as e:
            self._notify_status(f"Frame camera connection failed: {e}")
            return False
    
    def disconnect_camera(self):
        """Disconnect camera gracefully"""
        self.force_disconnect()
        self._notify_status("Frame camera disconnected")
    
    def force_disconnect(self):
        """Force disconnect with brutal cleanup"""
        try:
            # Stop all operations first
            self.stop_grabbing()
            
            # Give threads time to finish
            time.sleep(0.1)
            
            # Force close camera
            if self.cam:
                try:
                    self.cam.MV_CC_CloseDevice()
                except:
                    pass
                try:
                    self.cam.MV_CC_DestroyHandle()
                except:
                    pass
                self.cam = None
            
            # Close OpenCV writer if open
            if self.opencv_writer:
                try:
                    self.opencv_writer.release()
                except:
                    pass
                self.opencv_writer = None
            
            # Reset all flags
            self.device_info = None
            self.is_recording = False
            self.is_grabbing = False
            self.show_preview = False
            self.exit_flag = True
            self.use_opencv_recording = False
            
            # Clear queues
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except:
                    break
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break
                    
        except Exception as e:
            print(f"Error in force disconnect: {e}")
    
    def _setup_recording_params(self):
        """Setup recording parameters"""
        if not self.cam:
            return
            
        record_param = MV_CC_RECORD_PARAM()
        memset(byref(record_param), 0, sizeof(MV_CC_RECORD_PARAM))
        
        # Get camera parameters
        st_param = MVCC_INTVALUE()
        memset(byref(st_param), 0, sizeof(MVCC_INTVALUE))
        
        # Width
        ret = self.cam.MV_CC_GetIntValue("Width", st_param)
        if ret == 0:
            record_param.nWidth = st_param.nCurValue
        
        # Height
        ret = self.cam.MV_CC_GetIntValue("Height", st_param)
        if ret == 0:
            record_param.nHeight = st_param.nCurValue
        
        # Pixel format
        st_enum_value = MVCC_ENUMVALUE()
        memset(byref(st_enum_value), 0, sizeof(MVCC_ENUMVALUE))
        ret = self.cam.MV_CC_GetEnumValue("PixelFormat", st_enum_value)
        if ret == 0:
            record_param.enPixelType = MvGvspPixelType(st_enum_value.nCurValue)
        
        # Frame rate
        st_float_value = MVCC_FLOATVALUE()
        memset(byref(st_float_value), 0, sizeof(MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue("ResultingFrameRate", st_float_value)
        if ret != 0:
            ret = self.cam.MV_CC_GetFloatValue("AcquisitionFrameRate", st_float_value)
            if ret != 0:
                st_float_value.fCurValue = 30.0
        record_param.fFrameRate = st_float_value.fCurValue
        
        record_param.nBitRate = 5000
        record_param.enRecordFmtType = MV_FormatType_AVI
        
        # Setup RGB buffer
        self.rgb_buffer_size = record_param.nWidth * record_param.nHeight * 3
        self.rgb_buffer = (c_ubyte * self.rgb_buffer_size)()
        
        self.record_params = record_param
    
    def start_grabbing(self):
        """Start grabbing frames"""
        if not self.cam or self.is_grabbing:
            return self.is_grabbing
            
        ret = self.cam.MV_CC_StartGrabbing()
        if ret == 0:
            self.is_grabbing = True
            self.exit_flag = False
            self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
            self.recording_thread.start()
            self._notify_status("Frame grabbing started")
            return True
        else:
            self._notify_status(f"Failed to start grabbing: 0x{ret:08x}")
            return False
    
    def stop_grabbing(self):
        """Stop grabbing frames"""
        if self.is_grabbing:
            if self.is_recording:
                self.stop_recording()
            
            self.stop_preview()
            self.exit_flag = True
            
            if self.recording_thread:
                self.recording_thread.join(timeout=2.0)
            
            try:
                self.cam.MV_CC_StopGrabbing()
            except:
                pass
            self.is_grabbing = False
            self._notify_status("Frame grabbing stopped")
    
    def start_recording(self, filename=None):
        """Start recording"""
        # Auto-start grabbing if needed
        if not self.is_grabbing:
            if not self.start_grabbing():
                return False
        
        if not filename:
            device_name = self._get_device_name(self.device_info).replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{device_name}_{timestamp}.avi"
        
        self.current_filename = filename
        
        # Determine recording method based on transformations
        has_transformations = (self.vertical_flip or self.horizontal_flip or self.rotation_angle != 0)
        
        if has_transformations:
            # Use OpenCV recording for transformed frames
            self.use_opencv_recording = True
            self.command_queue.put(('START_OPENCV_RECORDING', filename))
        else:
            # Use SDK recording for non-transformed frames
            self.use_opencv_recording = False
            self.command_queue.put(('START_RECORDING', filename))
        
        return True
    
    def stop_recording(self):
        """Stop recording"""
        if self.use_opencv_recording:
            self.command_queue.put(('STOP_OPENCV_RECORDING',))
        else:
            self.command_queue.put(('STOP_RECORDING',))
    
    def start_preview(self):
        """Start preview with auto-start grabbing"""
        # Auto-start grabbing if needed
        if not self.is_grabbing:
            if not self.start_grabbing():
                return False
            
        if not self.show_preview:
            self.show_preview = True
            self.preview_thread = threading.Thread(target=self._preview_worker, daemon=True)
            self.preview_thread.start()
            return True
        return False
    
    def stop_preview(self):
        """Stop preview"""
        if self.show_preview:
            self.show_preview = False
            if self.preview_thread:
                self.preview_thread.join(timeout=2.0)
    
    def toggle_preview_always_on_top(self):
        """Toggle always on top for preview window"""
        self.preview_always_on_top = not self.preview_always_on_top
        if WINDOWS_AVAILABLE:
            set_window_always_on_top("Frame Camera Preview", self.preview_always_on_top)
        self._notify_status(f"Frame preview always on top: {'ON' if self.preview_always_on_top else 'OFF'}")
    
    def toggle_vertical_flip(self):
        """Toggle vertical flip"""
        self.vertical_flip = not self.vertical_flip
        self._notify_status(f"Frame vertical flip: {'ON' if self.vertical_flip else 'OFF'}")
        return self.vertical_flip
    
    def toggle_horizontal_flip(self):
        """Toggle horizontal flip"""
        self.horizontal_flip = not self.horizontal_flip
        self._notify_status(f"Frame horizontal flip: {'ON' if self.horizontal_flip else 'OFF'}")
        return self.horizontal_flip
    
    def rotate_frame(self):
        """Rotate frame by 90 degrees clockwise"""
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self._notify_status(f"Frame rotation: {self.rotation_angle}°")
        return self.rotation_angle
    
    def _apply_transformations(self, img):
        """Apply transformations to image"""
        if img is None:
            return img
        
        # Apply vertical flip
        if self.vertical_flip:
            img = cv2.flip(img, 0)
        
        # Apply horizontal flip
        if self.horizontal_flip:
            img = cv2.flip(img, 1)
        
        # Apply rotation
        if self.rotation_angle != 0:
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            
            if self.rotation_angle == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation_angle == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif self.rotation_angle == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return img
    
    def get_exposure_range(self):
        """Get exposure time range"""
        if not self.cam:
            return None
        
        st_float = MVCC_FLOATVALUE()
        memset(byref(st_float), 0, sizeof(MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", st_float)
        if ret == 0:
            return (st_float.fMin, st_float.fMax, st_float.fCurValue)
        return None
    
    def set_exposure(self, value):
        """Set exposure time"""
        if not self.cam:
            return False
        
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(value))
        return ret == 0
    
    def get_gain_range(self):
        """Get gain range"""
        if not self.cam:
            return None
        
        st_float = MVCC_FLOATVALUE()
        memset(byref(st_float), 0, sizeof(MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue("Gain", st_float)
        if ret == 0:
            return (st_float.fMin, st_float.fMax, st_float.fCurValue)
        return None
    
    def set_gain(self, value):
        """Set gain"""
        if not self.cam:
            return False
        
        self.cam.MV_CC_SetEnumValue("GainAuto", 0)
        ret = self.cam.MV_CC_SetFloatValue("Gain", float(value))
        return ret == 0
    
    def _recording_worker(self):
        """Recording worker thread"""
        frame_out = MV_FRAME_OUT()
        memset(byref(frame_out), 0, sizeof(frame_out))
        
        input_frame_info = MV_CC_INPUT_FRAME_INFO()
        memset(byref(input_frame_info), 0, sizeof(MV_CC_INPUT_FRAME_INFO))
        
        while not self.exit_flag:
            # Process commands
            try:
                cmd_data = self.command_queue.get_nowait()
                cmd = cmd_data[0]
                
                if cmd == 'START_RECORDING':
                    filename = cmd_data[1]
                    if not self.is_recording and self.is_grabbing:
                        self.record_params.strFilePath = filename.encode('ascii')
                        ret = self.cam.MV_CC_StartRecord(self.record_params)
                        if ret == 0:
                            self.is_recording = True
                            with self.stats_lock:
                                self.stats['frames_captured'] = 0
                                self.stats['recording_start_time'] = time.time()
                            self._notify_status(f"Recording started: {filename}")
                
                elif cmd == 'STOP_RECORDING':
                    if self.is_recording:
                        ret = self.cam.MV_CC_StopRecord()
                        if ret == 0:
                            self.is_recording = False
                            self._notify_status("Recording stopped")
                
                elif cmd == 'START_OPENCV_RECORDING':
                    filename = cmd_data[1]
                    if not self.is_recording and self.is_grabbing and self.record_params:
                        # Setup OpenCV VideoWriter
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        fps = self.record_params.fFrameRate
                        
                        # Get frame size (considering rotation)
                        width = self.record_params.nWidth
                        height = self.record_params.nHeight
                        if self.rotation_angle in [90, 270]:
                            width, height = height, width
                        
                        self.opencv_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                        if self.opencv_writer.isOpened():
                            self.is_recording = True
                            with self.stats_lock:
                                self.stats['frames_captured'] = 0
                                self.stats['recording_start_time'] = time.time()
                            self._notify_status(f"OpenCV recording started: {filename}")
                        else:
                            self._notify_status("Failed to start OpenCV recording")
                
                elif cmd == 'STOP_OPENCV_RECORDING':
                    if self.is_recording and self.opencv_writer:
                        self.opencv_writer.release()
                        self.opencv_writer = None
                        self.is_recording = False
                        self._notify_status("OpenCV recording stopped")
                        
            except queue.Empty:
                pass
            
            # Capture frames
            if self.is_grabbing:
                ret = self.cam.MV_CC_GetImageBuffer(frame_out, 100)
                if frame_out.pBufAddr and ret == 0:
                    # Convert frame for processing
                    img = self._convert_to_display_format(frame_out.stFrameInfo, frame_out.pBufAddr)
                    
                    if img is not None:
                        # Apply transformations
                        transformed_img = self._apply_transformations(img)
                        
                        # Handle recording
                        if self.is_recording:
                            if self.use_opencv_recording and self.opencv_writer and transformed_img is not None:
                                # Record transformed frame with OpenCV
                                self.opencv_writer.write(transformed_img)
                                with self.stats_lock:
                                    self.stats['frames_captured'] += 1
                            elif not self.use_opencv_recording:
                                # Record original frame with SDK
                                input_frame_info.pData = cast(frame_out.pBufAddr, POINTER(c_ubyte))
                                input_frame_info.nDataLen = frame_out.stFrameInfo.nFrameLen
                                ret = self.cam.MV_CC_InputOneFrame(input_frame_info)
                                if ret == 0:
                                    with self.stats_lock:
                                        self.stats['frames_captured'] += 1
                        
                        # Send to preview (always transformed)
                        if self.show_preview and transformed_img is not None:
                            try:
                                self.frame_queue.put_nowait(transformed_img)
                            except queue.Full:
                                pass
                    
                    self.cam.MV_CC_FreeImageBuffer(frame_out)
            else:
                time.sleep(0.01)
    
    def _preview_worker(self):
        """Preview worker thread"""
        window_name = "Frame Camera Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Position window on TOP HALF of right side of screen
        if self.screen_info:
            window_width = self.screen_info.get('right_half_width', 640)
            window_height = self.screen_info.get('top_half_height', 360) - 50  # Leave space for title bar
            window_x = self.screen_info.get('right_half_x', 700)
            window_y = 0  # TOP of screen
            
            # Resize and position the window
            cv2.resizeWindow(window_name, window_width, window_height)
            cv2.moveWindow(window_name, window_x, window_y)
        
        # Set initial always on top state - ALWAYS SET BY DEFAULT
        if WINDOWS_AVAILABLE:
            # Give window time to be created
            time.sleep(0.1)
            set_window_always_on_top(window_name, self.preview_always_on_top)
        
        # Enable mouse callback for right-click menu
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_RBUTTONDOWN:
                # Create a simple menu using messagebox
                self.toggle_preview_always_on_top()
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        self._notify_status("Frame preview controls: Right-click to toggle always on top, 'T' to toggle, ESC to close")
        
        while not self.exit_flag and self.show_preview:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Add overlay text showing transformation status
                overlay_frame = frame.copy()
                
                # Create status text
                transform_info = []
                if self.vertical_flip:
                    transform_info.append("V-Flip")
                if self.horizontal_flip:
                    transform_info.append("H-Flip")
                if self.rotation_angle != 0:
                    transform_info.append(f"Rot{self.rotation_angle}°")
                
                transform_text = " | ".join(transform_info) if transform_info else "No transforms"
                aot_text = f"Always on top: {'ON' if self.preview_always_on_top else 'OFF'}"
                
                cv2.putText(overlay_frame, f"Transforms: {transform_text}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay_frame, f"{aot_text} (Right-click/T to toggle)", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow(window_name, overlay_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self.show_preview = False
                    break
                elif key == ord('t') or key == ord('T'):  # Toggle always on top
                    self.toggle_preview_always_on_top()
                    
            except queue.Empty:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.show_preview = False
                    break
            except Exception as e:
                break
        
        cv2.destroyWindow(window_name)
    
    def _convert_to_display_format(self, frame_info, raw_data):
        """Convert raw camera data to displayable format"""
        memset(byref(self.convert_param), 0, sizeof(self.convert_param))
        self.convert_param.nWidth = frame_info.nWidth
        self.convert_param.nHeight = frame_info.nHeight
        self.convert_param.pSrcData = raw_data
        self.convert_param.nSrcDataLen = frame_info.nFrameLen
        self.convert_param.enSrcPixelType = frame_info.enPixelType
        self.convert_param.enDstPixelType = PixelType_Gvsp_RGB8_Packed
        self.convert_param.pDstBuffer = self.rgb_buffer
        self.convert_param.nDstBufferSize = self.rgb_buffer_size
        
        ret = self.cam.MV_CC_ConvertPixelType(self.convert_param)
        if ret != 0:
            return None
        
        img_buff = np.frombuffer(self.rgb_buffer, dtype=np.uint8, count=self.convert_param.nDstLen)
        img = img_buff.reshape((frame_info.nHeight, frame_info.nWidth, 3))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img_bgr
    
    def _notify_status(self, message):
        """Notify status callback"""
        if self.status_callback:
            self.status_callback(f"Frame: {message}")
    
    def get_status(self):
        """Get current status"""
        with self.stats_lock:
            return {
                'connected': self.cam is not None,
                'grabbing': self.is_grabbing,
                'recording': self.is_recording,
                'previewing': self.show_preview,
                'frames_captured': self.stats['frames_captured'],
                'fps': self.stats['last_fps'],
                'vertical_flip': self.vertical_flip,
                'horizontal_flip': self.horizontal_flip,
                'rotation_angle': self.rotation_angle
            }


class EventCameraController:
    """Modified version of EventCameraRecorder for GUI integration - simplified connection like old working code"""
    
    def __init__(self, status_callback=None, screen_info=None):
        self.device = None
        self.mv_iterator = None
        self.height = None
        self.width = None
        
        self.is_recording = False
        self.should_exit = False
        self.visualization_running = False
        self.capturing = False
        
        self.record_lock = threading.Lock()
        self.event_queue = queue.Queue(maxsize=1000)
        self.status_callback = status_callback
        self.screen_info = screen_info or {}
        
        self.current_log_path = None
        
        # Threads
        self.event_thread = None
        self.viz_thread = None
        
        # Window reference for proper cleanup and always on top - DEFAULT TO TRUE
        self.window = None
        self.window_always_on_top = True
    
    def find_cameras(self):
        """Find available event cameras - simplified approach"""
        if not EVENT_CAMERA_AVAILABLE:
            return []
        
        # Don't actually try to enumerate - just return a placeholder
        # The actual connection test will happen in connect_camera()
        return [(0, "Event Camera (Auto-detect)", "")]
    
    def connect_camera(self, device_path=""):
        """Connect to event camera - using simple approach from old working code"""
        if not EVENT_CAMERA_AVAILABLE:
            self._notify_status("Event camera SDK not available")
            return False
        
        try:
            self._notify_status("Connecting to event camera...")
            
            # Simple cleanup first
            self.stop_all()
            
            # Use the exact same approach as the old working code
            self.device = initiate_device("")
            
            if self.device is None:
                self._notify_status("Event camera connection failed: initiate_device returned None")
                return False
            
            # Test the connection by creating iterator - exactly like old code
            self.mv_iterator = EventsIterator.from_device(device=self.device, delta_t=1000)
            self.height, self.width = self.mv_iterator.get_size()
            
            self._notify_status(f"Event camera connected successfully: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            self._notify_status(f"Event camera connection failed: {e}")
            # Clean up failed attempt
            try:
                if self.device:
                    self.device = None
                self.mv_iterator = None
            except:
                pass
            return False
    
    def disconnect_camera(self):
        """Disconnect camera gracefully"""
        self.stop_all()
        self.device = None
        self.mv_iterator = None
        self._notify_status("Event camera disconnected")
    
    def start_capture(self):
        """Start capturing events"""
        if not self.device or self.capturing:
            return self.capturing
        
        self.should_exit = False
        self.capturing = True
        self.event_thread = threading.Thread(target=self._event_processing_thread, daemon=True)
        self.event_thread.start()
        self._notify_status("Event capture started")
        return True
    
    def stop_capture(self):
        """Stop capturing events"""
        if self.capturing:
            self.should_exit = True
            self.capturing = False
            if self.event_thread:
                self.event_thread.join(timeout=2.0)
            self._notify_status("Event capture stopped")
    
    def start_recording(self, output_dir="", filename_prefix=""):
        """Start recording events"""
        # Auto-start capture if needed
        if not self.capturing:
            if not self.start_capture():
                return False
        
        with self.record_lock:
            if self.is_recording:
                return False
            
            if self.device and self.device.get_i_events_stream():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if filename_prefix:
                    self.current_log_path = f"{filename_prefix}_{timestamp}.raw"
                else:
                    self.current_log_path = f"recording_{timestamp}.raw"
                
                if output_dir:
                    self.current_log_path = os.path.join(output_dir, self.current_log_path)
                
                self.device.get_i_events_stream().log_raw_data(self.current_log_path)
                self.is_recording = True
                self._notify_status(f"Recording started: {self.current_log_path}")
                return True
        return False
    
    def stop_recording(self):
        """Stop recording events"""
        with self.record_lock:
            if not self.is_recording:
                return False
            
            try:
                self.device.get_i_events_stream().stop_log_raw_data()
            except:
                pass
            self.is_recording = False
            self._notify_status(f"Recording stopped: {self.current_log_path}")
            return True
    
    def start_visualization(self):
        """Start event visualization with auto-start capture"""
        # Auto-start capture if needed
        if not self.capturing:
            if not self.start_capture():
                return False
        
        if not self.device or self.visualization_running:
            return False
        
        self.viz_thread = threading.Thread(target=self._visualization_thread, daemon=True)
        self.viz_thread.start()
        return True
    
    def stop_visualization(self):
        """Stop visualization"""
        if self.visualization_running:
            self.visualization_running = False
            
            # Force close window
            if self.window:
                try:
                    self.window.set_close_flag()
                except:
                    pass
                self.window = None
            
            if self.viz_thread:
                self.viz_thread.join(timeout=3.0)
    
    def toggle_visualization_always_on_top(self):
        """Toggle always on top for visualization window"""
        self.window_always_on_top = not self.window_always_on_top
        if WINDOWS_AVAILABLE:
            set_window_always_on_top("Event Camera Preview", self.window_always_on_top)
        self._notify_status(f"Event preview always on top: {'ON' if self.window_always_on_top else 'OFF'}")
    
    def stop_all(self):
        """Stop all operations"""
        if self.is_recording:
            self.stop_recording()
        self.stop_visualization()
        self.stop_capture()
    
    def get_bias_values(self):
        """Get current bias values"""
        if not self.device:
            return {}
        
        bias_interface = self.device.get_i_ll_biases()
        if bias_interface is None:
            return {}
        
        try:
            return bias_interface.get_all_biases()
        except:
            return {}
    
    def set_bias_value(self, bias_name, value):
        """Set a bias value"""
        if not self.device:
            return False
        
        bias_interface = self.device.get_i_ll_biases()
        if bias_interface is None:
            return False
        
        try:
            bias_interface.set(bias_name, int(value))
            return True
        except:
            return False
    
    def _event_processing_thread(self):
        """Thread for reading events from camera - exactly like old working code"""
        try:
            for evs in self.mv_iterator:
                if self.should_exit:
                    break
                
                try:
                    self.event_queue.put(evs, block=False)
                except queue.Full:
                    pass
                
                if self.should_exit:
                    break
        except Exception as e:
            self._notify_status(f"Event processing error: {e}")
        finally:
            self.capturing = False
    
    def _visualization_thread(self):
        """Thread for event visualization - like old working code but with always on top"""
        try:
            event_frame_gen = PeriodicFrameGenerationAlgorithm(
                sensor_width=self.width, sensor_height=self.height, 
                fps=25, palette=ColorPalette.Dark
            )
            
            # Calculate window dimensions for BOTTOM half of right side
            if self.screen_info:
                window_width = min(self.width, self.screen_info.get('right_half_width', self.width))
                # Calculate height to fit in bottom portion
                available_height = self.screen_info.get('height', 720) - self.screen_info.get('top_half_height', 360)
                window_height = min(self.height, available_height - 50)  # Leave space for title bar
            else:
                window_width = self.width
                window_height = self.height
            
            with MTWindow(title="Event Camera Preview", width=window_width, height=window_height,
                         mode=BaseWindow.RenderMode.BGR) as window:
                
                self.window = window
                
                # Position window on BOTTOM half of right side using Windows API
                if WINDOWS_AVAILABLE and self.screen_info:
                    def position_window():
                        time.sleep(0.3)  # Give window time to be created
                        try:
                            hwnd = win32gui.FindWindow(None, "Event Camera Preview")
                            if hwnd:
                                x = self.screen_info.get('right_half_x', 700)
                                # Position right after the top window with minimal gap
                                top_window_height = self.screen_info.get('top_half_height', 360)
                                y = top_window_height - 20  # Small gap between windows
                                width = self.screen_info.get('right_half_width', 640)
                                # Calculate height to fit remaining space
                                remaining_height = self.screen_info.get('height', 720) - y - 50  # Leave space at bottom
                                height = min(window_height, remaining_height)
                                
                                win32gui.SetWindowPos(hwnd, 0, x, y, width, height, 
                                                    win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE)
                        except Exception as e:
                            print(f"Could not position event window: {e}")
                    
                    # Position window in a separate thread to avoid blocking
                    import threading
                    threading.Thread(target=position_window, daemon=True).start()
                
                def keyboard_cb(key, scancode, action, mods):
                    if action != UIAction.RELEASE:
                        return
                    if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                        window.set_close_flag()
                    elif key == UIKeyEvent.KEY_T:
                        # Toggle always on top with 'T' key
                        self.toggle_visualization_always_on_top()
                
                window.set_keyboard_callback(keyboard_cb)
                
                def on_cd_frame_cb(ts, cd_frame):
                    # Add status overlay to the frame
                    overlay_frame = cd_frame.copy()
                    status_text = f"Always on top: {'ON' if self.window_always_on_top else 'OFF'} (T to toggle)"
                    cv2.putText(overlay_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    window.show_async(overlay_frame)
                
                event_frame_gen.set_output_callback(on_cd_frame_cb)
                
                self.visualization_running = True
                self._notify_status("Event visualization started - Press 'T' to toggle always on top")
                
                # Set initial always on top state - ALWAYS SET BY DEFAULT
                if WINDOWS_AVAILABLE:
                    time.sleep(0.4)  # Give window time to be created and positioned
                    set_window_always_on_top("Event Camera Preview", self.window_always_on_top)
                
                # Simple event loop like old working code
                while not self.should_exit and self.visualization_running:
                    if window.should_close():
                        break
                    
                    try:
                        evs = self.event_queue.get(timeout=0.01)
                        EventLoop.poll_and_dispatch()
                        event_frame_gen.process_events(evs)
                    except queue.Empty:
                        EventLoop.poll_and_dispatch()
                    except Exception as e:
                        break
            
        except Exception as e:
            self._notify_status(f"Visualization error: {e}")
        finally:
            self.visualization_running = False
            self.window = None
            self._notify_status("Event visualization stopped")
    
    def _notify_status(self, message):
        """Notify status callback"""
        if self.status_callback:
            self.status_callback(f"Event: {message}")
    
    def get_status(self):
        """Get current status"""
        return {
            'connected': self.device is not None,
            'capturing': self.capturing,
            'recording': self.is_recording,
            'visualizing': self.visualization_running
        }


class DualCameraGUI:
    """Main GUI application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dual Camera Control System")
        
        # Get screen dimensions for optimal window positioning
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Position main window on left half of screen
        main_width = screen_width - 20
        main_height = screen_height - 90  # Leave space for taskbar
        self.root.geometry(f"{main_width}x{main_height}+0+0")
        
        # Store screen info for camera preview positioning - UPDATED calculations
        self.screen_info = {
            'width': screen_width,
            'height': screen_height,
            'right_half_x': screen_width // 2,
            'right_half_width': screen_width // 2,
            'top_half_height': (screen_height - 100) // 2,  # Account for taskbar, split in half
            'bottom_half_y': (screen_height - 100) // 2 - 20  # Position for bottom window with small gap
        }
        
        # Camera controllers with screen info
        self.frame_camera = FrameCameraController(self.update_status, self.screen_info)
        self.event_camera = EventCameraController(self.update_status, self.screen_info)
        
        # GUI variables
        self.frame_cameras_list = []
        self.event_cameras_list = []
        
        # Status update queue
        self.status_queue = queue.Queue()
        
        # Parameter variables for text boxes
        self.exposure_text_var = tk.StringVar()
        self.gain_text_var = tk.StringVar()
        self.bias_text_vars = {}
        
        # Filename prefix variable
        self.filename_prefix_var = tk.StringVar()
        self.filename_prefix_var.set("sync_recording")
        
        # Unified control flags
        self.unified_preview_active = False
        self.unified_recording_active = False
        
        self.create_widgets()
        self.setup_status_update()
        
        # Auto-scan for cameras
        self.scan_cameras()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Main container with scrollable frame
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Filename prefix section
        prefix_section = ttk.LabelFrame(scrollable_frame, text="Recording Settings")
        prefix_section.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(prefix_section, text="Filename Prefix:").grid(row=0, column=0, padx=5, pady=5)
        prefix_entry = ttk.Entry(prefix_section, textvariable=self.filename_prefix_var, width=30)
        prefix_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        prefix_section.columnconfigure(1, weight=1)
        
        # Frame camera section
        frame_section = ttk.LabelFrame(scrollable_frame, text="Frame Camera")
        frame_section.pack(fill=tk.X, padx=5, pady=5)
        self.create_frame_camera_section(frame_section)
        
        # Event camera section
        event_section = ttk.LabelFrame(scrollable_frame, text="Event Camera")
        event_section.pack(fill=tk.X, padx=5, pady=5)
        self.create_event_camera_section(event_section)
        
        # Unified controls section
        unified_section = ttk.LabelFrame(scrollable_frame, text="Unified Controls")
        unified_section.pack(fill=tk.X, padx=5, pady=5)
        self.create_unified_controls_section(unified_section)
        
        # Always on top controls section
        aot_section = ttk.LabelFrame(scrollable_frame, text="Preview Window Controls")
        aot_section.pack(fill=tk.X, padx=5, pady=5)
        self.create_always_on_top_section(aot_section)
        
        # Status section
        status_section = ttk.LabelFrame(scrollable_frame, text="Status Log")
        status_section.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.create_status_section(status_section)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_frame_camera_section(self, parent):
        """Create frame camera control section"""
        # Connection frame
        conn_frame = ttk.LabelFrame(parent, text="Connection")
        conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(conn_frame, text="Camera:").grid(row=0, column=0, padx=5, pady=5)
        self.frame_camera_var = tk.StringVar()
        self.frame_camera_combo = ttk.Combobox(conn_frame, textvariable=self.frame_camera_var, state="readonly")
        self.frame_camera_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Button(conn_frame, text="Scan", command=self.scan_cameras).grid(row=0, column=2, padx=5, pady=5)
        self.frame_connect_btn = ttk.Button(conn_frame, text="Connect", command=self.connect_frame_camera)
        self.frame_connect_btn.grid(row=0, column=3, padx=5, pady=5)
        
        conn_frame.columnconfigure(1, weight=1)
        
        # Control frame
        control_frame = ttk.LabelFrame(parent, text="Individual Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.frame_grab_btn = ttk.Button(control_frame, text="Start Grabbing", command=self.toggle_frame_grabbing)
        self.frame_grab_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.frame_preview_btn = ttk.Button(control_frame, text="Start Preview", command=self.toggle_frame_preview)
        self.frame_preview_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.frame_record_btn = ttk.Button(control_frame, text="Start Recording", command=self.toggle_frame_recording)
        self.frame_record_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Frame transformation controls
        transform_frame = ttk.LabelFrame(parent, text="Frame Transformations (Default: V-Flip ON)")
        transform_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.vflip_btn = ttk.Button(transform_frame, text="Toggle Vertical Flip [ON]", 
                                   command=self.toggle_vertical_flip, width=20)
        self.vflip_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.hflip_btn = ttk.Button(transform_frame, text="Toggle Horizontal Flip [OFF]", 
                                   command=self.toggle_horizontal_flip, width=20)
        self.hflip_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.rotate_btn = ttk.Button(transform_frame, text="Rotate 90° [0°]", 
                                    command=self.rotate_frame, width=15)
        self.rotate_btn.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(transform_frame, text="Note: Transforms apply to both preview and recording", 
                 font=('TkDefaultFont', 8)).grid(row=1, column=0, columnspan=3, padx=5, pady=2)
        
        # Parameters frame
        param_frame = ttk.LabelFrame(parent, text="Parameters")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Exposure
        ttk.Label(param_frame, text="Exposure (μs):").grid(row=0, column=0, padx=5, pady=5)
        self.exposure_var = tk.DoubleVar()
        self.exposure_scale = ttk.Scale(param_frame, from_=100, to=10000, variable=self.exposure_var, 
                                       orient=tk.HORIZONTAL, command=self.on_exposure_scale_change)
        self.exposure_scale.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.exposure_entry = ttk.Entry(param_frame, textvariable=self.exposure_text_var, width=10)
        self.exposure_entry.grid(row=0, column=2, padx=5, pady=5)
        self.exposure_entry.bind('<Return>', self.on_exposure_entry_change)
        self.exposure_entry.bind('<FocusOut>', self.on_exposure_entry_change)
        
        # Gain
        ttk.Label(param_frame, text="Gain (dB):").grid(row=1, column=0, padx=5, pady=5)
        self.gain_var = tk.DoubleVar()
        self.gain_scale = ttk.Scale(param_frame, from_=0, to=20, variable=self.gain_var,
                                   orient=tk.HORIZONTAL, command=self.on_gain_scale_change)
        self.gain_scale.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        self.gain_entry = ttk.Entry(param_frame, textvariable=self.gain_text_var, width=10)
        self.gain_entry.grid(row=1, column=2, padx=5, pady=5)
        self.gain_entry.bind('<Return>', self.on_gain_entry_change)
        self.gain_entry.bind('<FocusOut>', self.on_gain_entry_change)
        
        param_frame.columnconfigure(1, weight=1)
    
    def create_event_camera_section(self, parent):
        """Create event camera control section"""
        # Connection frame
        conn_frame = ttk.LabelFrame(parent, text="Connection")
        conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(conn_frame, text="Camera:").grid(row=0, column=0, padx=5, pady=5)
        self.event_camera_var = tk.StringVar()
        self.event_camera_combo = ttk.Combobox(conn_frame, textvariable=self.event_camera_var, state="readonly")
        self.event_camera_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.event_connect_btn = ttk.Button(conn_frame, text="Connect", command=self.connect_event_camera)
        self.event_connect_btn.grid(row=0, column=2, padx=5, pady=5)
        
        conn_frame.columnconfigure(1, weight=1)
        
        # Control frame
        control_frame = ttk.LabelFrame(parent, text="Individual Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.event_capture_btn = ttk.Button(control_frame, text="Start Capture", command=self.toggle_event_capture)
        self.event_capture_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.event_viz_btn = ttk.Button(control_frame, text="Start Visualization", command=self.toggle_event_visualization)
        self.event_viz_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.event_record_btn = ttk.Button(control_frame, text="Start Recording", command=self.toggle_event_recording)
        self.event_record_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Bias parameters frame
        bias_frame = ttk.LabelFrame(parent, text="Bias Parameters")
        bias_frame.pack(fill=tk.X, padx=5, pady=5)
        
        bias_names = ['bias_diff', 'bias_diff_off', 'bias_diff_on', 'bias_fo', 'bias_hpf', 'bias_refr']
        self.bias_vars = {}
        self.bias_scales = {}
        self.bias_entries = {}
        
        for i, bias_name in enumerate(bias_names):
            row = i // 2
            col = (i % 2) * 4
            
            ttk.Label(bias_frame, text=f"{bias_name}:").grid(row=row, column=col, padx=5, pady=2)
            var = tk.IntVar()
            scale = ttk.Scale(bias_frame, from_=0, to=255, variable=var, orient=tk.HORIZONTAL,
                             command=lambda val, name=bias_name: self.on_bias_scale_change(name, val))
            scale.grid(row=row, column=col+1, padx=5, pady=2, sticky="ew")
            
            # Text entry for bias
            text_var = tk.StringVar()
            entry = ttk.Entry(bias_frame, textvariable=text_var, width=8)
            entry.grid(row=row, column=col+2, padx=5, pady=2)
            entry.bind('<Return>', lambda e, name=bias_name: self.on_bias_entry_change(name))
            entry.bind('<FocusOut>', lambda e, name=bias_name: self.on_bias_entry_change(name))
            
            self.bias_vars[bias_name] = var
            self.bias_scales[bias_name] = scale
            self.bias_text_vars[bias_name] = text_var
            self.bias_entries[bias_name] = entry
        
        bias_frame.columnconfigure(1, weight=1)
        bias_frame.columnconfigure(5, weight=1)
    
    def create_unified_controls_section(self, parent):
        """Create unified controls section"""
        ttk.Label(parent, text="Control both cameras simultaneously:", 
                 font=('TkDefaultFont', 10, 'bold')).pack(padx=5, pady=5)
        
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(padx=5, pady=5)
        
        self.unified_preview_btn = ttk.Button(controls_frame, text="Start Unified Preview", 
                                             command=self.toggle_unified_preview)
        self.unified_preview_btn.grid(row=0, column=0, padx=10, pady=5)
        
        self.unified_record_btn = ttk.Button(controls_frame, text="Start Unified Recording", 
                                            command=self.toggle_unified_recording)
        self.unified_record_btn.grid(row=0, column=1, padx=10, pady=5)
    
    def create_always_on_top_section(self, parent):
        """Create always on top controls section"""
        info_text = "Preview Window Controls (Default: Always On Top Enabled):"
        if not WINDOWS_AVAILABLE:
            info_text += " (Windows only - install pywin32 for full support)"
        
        ttk.Label(parent, text=info_text, font=('TkDefaultFont', 9)).pack(padx=5, pady=2)
        
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Toggle Frame Preview Always On Top", 
                  command=self.frame_camera.toggle_preview_always_on_top).grid(row=0, column=0, padx=5, pady=2)
        
        ttk.Button(controls_frame, text="Toggle Event Preview Always On Top", 
                  command=self.event_camera.toggle_visualization_always_on_top).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(parent, text="• Frame Preview: Right-click or press 'T' in preview window\n• Event Preview: Press 'T' in preview window\n• Frame: TOP half, Event: BOTTOM half", 
                 font=('TkDefaultFont', 8)).pack(padx=5, pady=2)
    
    def create_status_section(self, parent):
        """Create status section"""
        self.status_text = tk.Text(parent, height=12, state=tk.DISABLED)
        status_scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scroll.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def scan_cameras(self):
        """Scan for available cameras"""
        # Frame cameras
        self.frame_cameras_list = self.frame_camera.find_cameras()
        frame_names = [f"{idx}: {name}" for idx, name, _ in self.frame_cameras_list]
        self.frame_camera_combo['values'] = frame_names
        if frame_names:
            self.frame_camera_combo.current(0)
        
        # Event cameras - simplified approach
        self.event_cameras_list = self.event_camera.find_cameras()
        event_names = [f"{idx}: {name}" for idx, name, _ in self.event_cameras_list]
        self.event_camera_combo['values'] = event_names
        if event_names:
            self.event_camera_combo.current(0)
        
        # Update status
        frame_count = len(self.frame_cameras_list)
        event_count = len(self.event_cameras_list)
        self.update_status(f"Found {frame_count} frame cameras, {event_count} event cameras")
    
    def connect_frame_camera(self):
        """Connect to selected frame camera"""
        if not self.frame_cameras_list:
            messagebox.showerror("Error", "No frame cameras found!")
            return
        
        selected_idx = self.frame_camera_combo.current()
        if selected_idx < 0:
            messagebox.showerror("Error", "Please select a camera!")
            return
        
        idx, name, device_info = self.frame_cameras_list[selected_idx]
        
        if self.frame_camera.connect_camera(device_info):
            self.frame_connect_btn.config(text="Disconnect", command=self.disconnect_frame_camera)
            self.update_frame_parameters()
        else:
            messagebox.showerror("Error", "Failed to connect to frame camera!")
    
    def disconnect_frame_camera(self):
        """Disconnect frame camera"""
        self.frame_camera.disconnect_camera()
        self.frame_connect_btn.config(text="Connect", command=self.connect_frame_camera)
        self.reset_frame_button_states()
    
    def connect_event_camera(self):
        """Connect to event camera - simplified approach"""
        if self.event_camera.connect_camera():
            self.event_connect_btn.config(text="Disconnect", command=self.disconnect_event_camera)
            self.update_event_parameters()
        else:
            messagebox.showerror("Error", "Failed to connect to event camera!")
    
    def disconnect_event_camera(self):
        """Disconnect event camera"""
        self.event_camera.disconnect_camera()
        self.event_connect_btn.config(text="Connect", command=self.connect_event_camera)
        self.reset_event_button_states()
    
    def reset_frame_button_states(self):
        """Reset frame camera button states"""
        self.frame_grab_btn.config(text="Start Grabbing")
        self.frame_preview_btn.config(text="Start Preview")
        self.frame_record_btn.config(text="Start Recording")
    
    def reset_event_button_states(self):
        """Reset event camera button states"""
        self.event_capture_btn.config(text="Start Capture")
        self.event_viz_btn.config(text="Start Visualization")
        self.event_record_btn.config(text="Start Recording")
    
    def update_frame_parameters(self):
        """Update frame camera parameter ranges"""
        exposure_range = self.frame_camera.get_exposure_range()
        if exposure_range:
            min_exp, max_exp, cur_exp = exposure_range
            self.exposure_scale.config(from_=min_exp, to=max_exp)
            self.exposure_var.set(cur_exp)
            self.exposure_text_var.set(f"{cur_exp:.0f}")
        
        gain_range = self.frame_camera.get_gain_range()
        if gain_range:
            min_gain, max_gain, cur_gain = gain_range
            self.gain_scale.config(from_=min_gain, to=max_gain)
            self.gain_var.set(cur_gain)
            self.gain_text_var.set(f"{cur_gain:.1f}")
    
    def update_event_parameters(self):
        """Update event camera bias values"""
        bias_values = self.event_camera.get_bias_values()
        for bias_name, var in self.bias_vars.items():
            if bias_name in bias_values:
                value = bias_values[bias_name]
                var.set(value)
                self.bias_text_vars[bias_name].set(str(value))
    
    def toggle_frame_grabbing(self):
        """Toggle frame grabbing"""
        if not self.frame_camera.is_grabbing:
            if self.frame_camera.start_grabbing():
                self.frame_grab_btn.config(text="Stop Grabbing")
        else:
            self.frame_camera.stop_grabbing()
            self.reset_frame_button_states()
    
    def toggle_frame_preview(self):
        """Toggle frame preview with auto-start grabbing"""
        if not self.frame_camera.show_preview:
            if self.frame_camera.start_preview():
                self.frame_preview_btn.config(text="Stop Preview")
                # Update grabbing button if auto-started
                if self.frame_camera.is_grabbing:
                    self.frame_grab_btn.config(text="Stop Grabbing")
        else:
            self.frame_camera.stop_preview()
            self.frame_preview_btn.config(text="Start Preview")
    
    def toggle_frame_recording(self):
        """Toggle frame recording with auto-start grabbing"""
        if not self.frame_camera.is_recording:
            prefix = self.filename_prefix_var.get()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_frame_{timestamp}.avi" if prefix else None
            
            if self.frame_camera.start_recording(filename):
                self.frame_record_btn.config(text="Stop Recording")
                # Update grabbing button if auto-started
                if self.frame_camera.is_grabbing:
                    self.frame_grab_btn.config(text="Stop Grabbing")
        else:
            self.frame_camera.stop_recording()
            self.frame_record_btn.config(text="Start Recording")
    
    def toggle_vertical_flip(self):
        """Toggle vertical flip"""
        state = self.frame_camera.toggle_vertical_flip()
        self.vflip_btn.config(text=f"Toggle Vertical Flip [{'ON' if state else 'OFF'}]")
    
    def toggle_horizontal_flip(self):
        """Toggle horizontal flip"""
        state = self.frame_camera.toggle_horizontal_flip()
        self.hflip_btn.config(text=f"Toggle Horizontal Flip [{'ON' if state else 'OFF'}]")
    
    def rotate_frame(self):
        """Rotate frame by 90 degrees"""
        angle = self.frame_camera.rotate_frame()
        self.rotate_btn.config(text=f"Rotate 90° [{angle}°]")
    
    def toggle_event_capture(self):
        """Toggle event capture"""
        if not self.event_camera.capturing:
            if self.event_camera.start_capture():
                self.event_capture_btn.config(text="Stop Capture")
        else:
            self.event_camera.stop_capture()
            self.reset_event_button_states()
    
    def toggle_event_visualization(self):
        """Toggle event visualization with auto-start capture"""
        if not self.event_camera.visualization_running:
            if self.event_camera.start_visualization():
                self.event_viz_btn.config(text="Stop Visualization")
                # Update capture button if auto-started
                if self.event_camera.capturing:
                    self.event_capture_btn.config(text="Stop Capture")
        else:
            self.event_camera.stop_visualization()
            self.event_viz_btn.config(text="Start Visualization")
    
    def toggle_event_recording(self):
        """Toggle event recording with auto-start capture"""
        if not self.event_camera.is_recording:
            prefix = self.filename_prefix_var.get()
            if self.event_camera.start_recording(filename_prefix=prefix):
                self.event_record_btn.config(text="Stop Recording")
                # Update capture button if auto-started
                if self.event_camera.capturing:
                    self.event_capture_btn.config(text="Stop Capture")
        else:
            self.event_camera.stop_recording()
            self.event_record_btn.config(text="Start Recording")
    
    def toggle_unified_preview(self):
        """Toggle unified preview for both cameras"""
        if not self.unified_preview_active:
            # Start unified preview
            frame_started = False
            event_started = False
            
            # Start frame camera preview if connected
            if self.frame_camera.cam is not None:
                frame_started = self.frame_camera.start_preview()
                if frame_started:
                    self.frame_preview_btn.config(text="Stop Preview")
                    if self.frame_camera.is_grabbing:
                        self.frame_grab_btn.config(text="Stop Grabbing")
            
            # Start event camera preview if connected
            if self.event_camera.device is not None:
                event_started = self.event_camera.start_visualization()
                if event_started:
                    self.event_viz_btn.config(text="Stop Visualization")
                    if self.event_camera.capturing:
                        self.event_capture_btn.config(text="Stop Capture")
            
            if frame_started or event_started:
                self.unified_preview_active = True
                self.unified_preview_btn.config(text="Stop Unified Preview")
                self.update_status("Unified preview started")
            else:
                self.update_status("No cameras available for unified preview")
        else:
            # Stop unified preview
            if self.frame_camera.show_preview:
                self.frame_camera.stop_preview()
                self.frame_preview_btn.config(text="Start Preview")
            
            if self.event_camera.visualization_running:
                self.event_camera.stop_visualization()
                self.event_viz_btn.config(text="Start Visualization")
            
            self.unified_preview_active = False
            self.unified_preview_btn.config(text="Start Unified Preview")
            self.update_status("Unified preview stopped")
    
    def toggle_unified_recording(self):
        """Toggle unified recording for both cameras"""
        if not self.unified_recording_active:
            # Start unified recording
            frame_started = False
            event_started = False
            prefix = self.filename_prefix_var.get()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Start frame camera recording if connected
            if self.frame_camera.cam is not None:
                filename = f"{prefix}_frame_{timestamp}.avi" if prefix else None
                frame_started = self.frame_camera.start_recording(filename)
                if frame_started:
                    self.frame_record_btn.config(text="Stop Recording")
                    if self.frame_camera.is_grabbing:
                        self.frame_grab_btn.config(text="Stop Grabbing")
            
            # Start event camera recording if connected
            if self.event_camera.device is not None:
                event_started = self.event_camera.start_recording(filename_prefix=f"{prefix}_event" if prefix else "")
                if event_started:
                    self.event_record_btn.config(text="Stop Recording")
                    if self.event_camera.capturing:
                        self.event_capture_btn.config(text="Stop Capture")
            
            if frame_started or event_started:
                self.unified_recording_active = True
                self.unified_record_btn.config(text="Stop Unified Recording")
                self.update_status("Unified recording started")
            else:
                self.update_status("No cameras available for unified recording")
        else:
            # Stop unified recording
            if self.frame_camera.is_recording:
                self.frame_camera.stop_recording()
                self.frame_record_btn.config(text="Start Recording")
            
            if self.event_camera.is_recording:
                self.event_camera.stop_recording()
                self.event_record_btn.config(text="Start Recording")
            
            self.unified_recording_active = False
            self.unified_record_btn.config(text="Start Unified Recording")
            self.update_status("Unified recording stopped")
    
    def on_exposure_scale_change(self, value):
        """Handle exposure scale change"""
        exposure = float(value)
        self.exposure_text_var.set(f"{exposure:.0f}")
        if self.frame_camera.set_exposure(exposure):
            pass  # Success
    
    def on_exposure_entry_change(self, event=None):
        """Handle exposure text entry change"""
        try:
            exposure = float(self.exposure_text_var.get())
            # Get current range
            exposure_range = self.frame_camera.get_exposure_range()
            if exposure_range:
                min_exp, max_exp, _ = exposure_range
                # Clamp to valid range
                exposure = max(min_exp, min(max_exp, exposure))
                self.exposure_var.set(exposure)
                self.exposure_text_var.set(f"{exposure:.0f}")
                self.frame_camera.set_exposure(exposure)
        except ValueError:
            # Reset to current scale value
            self.exposure_text_var.set(f"{self.exposure_var.get():.0f}")
    
    def on_gain_scale_change(self, value):
        """Handle gain scale change"""
        gain = float(value)
        self.gain_text_var.set(f"{gain:.1f}")
        if self.frame_camera.set_gain(gain):
            pass  # Success
    
    def on_gain_entry_change(self, event=None):
        """Handle gain text entry change"""
        try:
            gain = float(self.gain_text_var.get())
            # Get current range
            gain_range = self.frame_camera.get_gain_range()
            if gain_range:
                min_gain, max_gain, _ = gain_range
                # Clamp to valid range
                gain = max(min_gain, min(max_gain, gain))
                self.gain_var.set(gain)
                self.gain_text_var.set(f"{gain:.1f}")
                self.frame_camera.set_gain(gain)
        except ValueError:
            # Reset to current scale value
            self.gain_text_var.set(f"{self.gain_var.get():.1f}")
    
    def on_bias_scale_change(self, bias_name, value):
        """Handle bias scale change"""
        bias_value = int(float(value))
        self.bias_text_vars[bias_name].set(str(bias_value))
        self.event_camera.set_bias_value(bias_name, bias_value)
    
    def on_bias_entry_change(self, bias_name):
        """Handle bias text entry change"""
        try:
            bias_value = int(self.bias_text_vars[bias_name].get())
            # Clamp to valid range (0-255)
            bias_value = max(0, min(255, bias_value))
            self.bias_vars[bias_name].set(bias_value)
            self.bias_text_vars[bias_name].set(str(bias_value))
            self.event_camera.set_bias_value(bias_name, bias_value)
        except ValueError:
            # Reset to current scale value
            self.bias_text_vars[bias_name].set(str(self.bias_vars[bias_name].get()))
    
    def update_status(self, message):
        """Update status message"""
        self.status_queue.put(message)
    
    def setup_status_update(self):
        """Setup periodic status updates"""
        def update():
            try:
                while True:
                    message = self.status_queue.get_nowait()
                    self.status_var.set(message)
                    
                    # Also add to status text
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    full_message = f"[{timestamp}] {message}\n"
                    
                    self.status_text.config(state=tk.NORMAL)
                    self.status_text.insert(tk.END, full_message)
                    self.status_text.see(tk.END)
                    self.status_text.config(state=tk.DISABLED)
            except queue.Empty:
                pass
            
            # Schedule next update
            self.root.after(100, update)
        
        # Start the update loop
        update()
    
    def run(self):
        """Run the GUI application"""
        def on_closing():
            # Cleanup with force disconnect
            try:
                self.frame_camera.force_disconnect()
                self.event_camera.stop_all()
                time.sleep(0.2)  # Give time for cleanup
            except:
                pass
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()


def main():
    """Main function"""
    app = DualCameraGUI()
    app.run()


if __name__ == "__main__":
    main()