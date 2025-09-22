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
from PIL import Image, ImageTk
import io

# Add the camera SDK paths (modify these paths according to your setup)
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, "haikang_sdk", "Python"))
sys.path.append(os.path.join(BASE, "haikang_sdk", "Python", "MvImport"))

# Import frame camera modules
try:
    from ctypes import *
    import numpy as np
    import cv2
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


def cv2_to_tkinter_optimized(cv_image, target_width, target_height, scale_factor=1.0):
    """Optimized conversion from OpenCV to tkinter PhotoImage"""
    if cv_image is None:
        return None
    
    try:
        # Calculate final dimensions based on scale factor
        final_width = int(target_width * scale_factor)
        final_height = int(target_height * scale_factor)
        
        # Resize image efficiently
        if final_width != cv_image.shape[1] or final_height != cv_image.shape[0]:
            cv_image = cv2.resize(cv_image, (final_width, final_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB more efficiently
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and then PhotoImage
        pil_image = Image.fromarray(rgb_image)
        return ImageTk.PhotoImage(pil_image)
    except Exception as e:
        print(f"Image conversion error: {e}")
        return None


class FrameCameraController:
    """Optimized Frame Camera Controller with better performance"""
    
    def __init__(self, status_callback=None, preview_widget=None, scale_callback=None):
        self.cam = None
        self.device_info = None
        self.recording_thread = None
        self.preview_thread = None
        self.command_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=15)  # Larger queue for better performance
        self.is_recording = False
        self.is_grabbing = False
        self.show_preview = False
        self.exit_flag = False
        self.current_filename = None
        self.record_params = None
        self.stats_lock = threading.Lock()
        self.status_callback = status_callback
        self.preview_widget = preview_widget
        self.scale_callback = scale_callback  # Callback to get current scale factor
        
        # Frame transformation settings
        self.vertical_flip = True
        self.horizontal_flip = False
        self.rotation_angle = 0
        self.use_opencv_recording = False
        self.opencv_writer = None
        
        # Performance optimization
        self.frame_skip_counter = 0
        self.preview_fps_limit = 30  # Limit preview FPS for performance
        self.last_preview_time = 0
        
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
            self.force_disconnect()
            
            try:
                MvCamera.MV_CC_Initialize()
            except:
                pass
            
            self.cam = MvCamera()
            self.device_info = device_info
            
            if self.cam.MV_CC_CreateHandle(device_info) != 0:
                return False
            
            if self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
                self.cam.MV_CC_DestroyHandle()
                return False
            
            if device_info.nTLayerType == MV_GIGE_DEVICE:
                pkt = self.cam.MV_CC_GetOptimalPacketSize()
                if pkt > 0:
                    self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt)
            
            self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
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
        """Force disconnect with cleanup"""
        try:
            self.stop_grabbing()
            time.sleep(0.1)
            
            if self.cam:
                try:
                    self.cam.MV_CC_CloseDevice()
                    self.cam.MV_CC_DestroyHandle()
                except:
                    pass
                self.cam = None
            
            if self.opencv_writer:
                try:
                    self.opencv_writer.release()
                except:
                    pass
                self.opencv_writer = None
            
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
        if not self.is_grabbing:
            if not self.start_grabbing():
                return False
        
        if not filename:
            device_name = self._get_device_name(self.device_info).replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{device_name}_{timestamp}.avi"
        
        self.current_filename = filename
        
        has_transformations = (self.vertical_flip or self.horizontal_flip or self.rotation_angle != 0)
        
        if has_transformations:
            self.use_opencv_recording = True
            self.command_queue.put(('START_OPENCV_RECORDING', filename))
        else:
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
            if self.preview_widget:
                self.preview_widget.configure(image='', text="Frame Camera\nNot Active")
    
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
        
        if self.vertical_flip:
            img = cv2.flip(img, 0)
        
        if self.horizontal_flip:
            img = cv2.flip(img, 1)
        
        if self.rotation_angle != 0:
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
        """Optimized recording worker thread"""
        frame_out = MV_FRAME_OUT()
        memset(byref(frame_out), 0, sizeof(frame_out))
        
        input_frame_info = MV_CC_INPUT_FRAME_INFO()
        memset(byref(input_frame_info), 0, sizeof(MV_CC_INPUT_FRAME_INFO))
        
        frame_count = 0
        
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
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        fps = self.record_params.fFrameRate
                        
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
                ret = self.cam.MV_CC_GetImageBuffer(frame_out, 50)  # Reduced timeout for better performance
                if frame_out.pBufAddr and ret == 0:
                    img = self._convert_to_display_format(frame_out.stFrameInfo, frame_out.pBufAddr)
                    
                    if img is not None:
                        transformed_img = self._apply_transformations(img)
                        
                        # Handle recording
                        if self.is_recording:
                            if self.use_opencv_recording and self.opencv_writer and transformed_img is not None:
                                self.opencv_writer.write(transformed_img)
                                with self.stats_lock:
                                    self.stats['frames_captured'] += 1
                            elif not self.use_opencv_recording:
                                input_frame_info.pData = cast(frame_out.pBufAddr, POINTER(c_ubyte))
                                input_frame_info.nDataLen = frame_out.stFrameInfo.nFrameLen
                                ret = self.cam.MV_CC_InputOneFrame(input_frame_info)
                                if ret == 0:
                                    with self.stats_lock:
                                        self.stats['frames_captured'] += 1
                        
                        # Send to preview with frame skipping for performance
                        if self.show_preview and transformed_img is not None:
                            frame_count += 1
                            # Skip frames if queue is getting full (keep only latest frames)
                            if frame_count % 2 == 0 or self.frame_queue.qsize() < 5:  # Skip every other frame if queue is filling up
                                try:
                                    # Clear old frames if queue is full
                                    if self.frame_queue.full():
                                        try:
                                            self.frame_queue.get_nowait()
                                        except:
                                            pass
                                    self.frame_queue.put_nowait(transformed_img)
                                except queue.Full:
                                    pass
                    
                    self.cam.MV_CC_FreeImageBuffer(frame_out)
            else:
                time.sleep(0.005)  # Shorter sleep for better responsiveness
    
    def _preview_worker(self):
        """Optimized preview worker thread"""
        self._notify_status("Frame preview started - embedded mode")
        
        frame_counter = 0
        last_update_time = time.time()
        
        while not self.exit_flag and self.show_preview:
            try:
                frame = self.frame_queue.get(timeout=0.05)  # Shorter timeout
                
                current_time = time.time()
                frame_counter += 1
                
                # Limit preview update rate for performance
                if current_time - last_update_time >= (1.0 / self.preview_fps_limit):
                    if self.preview_widget and frame is not None:
                        # Get current scale factor
                        scale_factor = 1.0
                        if self.scale_callback:
                            scale_factor = self.scale_callback()
                        
                        # Add overlay text less frequently for performance
                        if frame_counter % 10 == 0:  # Only add overlay every 10th frame
                            overlay_frame = frame.copy()
                            
                            transform_info = []
                            if self.vertical_flip:
                                transform_info.append("V-Flip")
                            if self.horizontal_flip:
                                transform_info.append("H-Flip")
                            if self.rotation_angle != 0:
                                transform_info.append(f"Rot{self.rotation_angle}°")
                            
                            transform_text = " | ".join(transform_info) if transform_info else "No transforms"
                            
                            cv2.putText(overlay_frame, f"Frame: {transform_text}", (10, 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            if self.is_recording:
                                cv2.putText(overlay_frame, "RECORDING", (10, 45), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        else:
                            overlay_frame = frame
                        
                        # Get widget dimensions
                        widget_width = self.preview_widget.winfo_width()
                        widget_height = self.preview_widget.winfo_height()
                        
                        if widget_width > 1 and widget_height > 1:  # Valid dimensions
                            # Convert to tkinter format with scale
                            tk_image = cv2_to_tkinter_optimized(overlay_frame, widget_width, widget_height, scale_factor)
                            if tk_image:
                                try:
                                    self.preview_widget.configure(image=tk_image, text="")
                                    self.preview_widget.image = tk_image
                                except:
                                    pass
                    
                    last_update_time = current_time
                    
            except queue.Empty:
                continue
            except Exception as e:
                break
        
        # Clear preview when stopped
        if self.preview_widget:
            try:
                self.preview_widget.configure(image='', text="Frame Camera\nNot Active")
                self.preview_widget.image = None
            except:
                pass
    
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
    """Optimized Event Camera Controller with better performance"""
    
    def __init__(self, status_callback=None, preview_widget=None, scale_callback=None):
        self.device = None
        self.mv_iterator = None
        self.height = None
        self.width = None
        
        self.is_recording = False
        self.should_exit = False
        self.visualization_running = False
        self.capturing = False
        
        self.record_lock = threading.Lock()
        self.event_queue = queue.Queue(maxsize=2000)  # Larger queue for events
        self.frame_queue = queue.Queue(maxsize=10)    # Separate frame queue for visualization
        self.status_callback = status_callback
        self.preview_widget = preview_widget
        self.scale_callback = scale_callback
        
        self.current_log_path = None
        
        # Threads
        self.event_thread = None
        self.viz_thread = None
        self.preview_thread = None
        
        # Frame generation for visualization
        self.event_frame_gen = None
        
        # Performance optimization
        self.event_fps_limit = 25  # Limit event visualization FPS
        self.last_viz_time = 0
    
    def find_cameras(self):
        """Find available event cameras"""
        if not EVENT_CAMERA_AVAILABLE:
            return []
        
        return [(0, "Event Camera (Auto-detect)", "")]
    
    def connect_camera(self, device_path=""):
        """Connect to event camera"""
        if not EVENT_CAMERA_AVAILABLE:
            self._notify_status("Event camera SDK not available")
            return False
        
        try:
            self._notify_status("Connecting to event camera...")
            
            self.stop_all()
            
            self.device = initiate_device("")
            
            if self.device is None:
                self._notify_status("Event camera connection failed: initiate_device returned None")
                return False
            
            self.mv_iterator = EventsIterator.from_device(device=self.device, delta_t=1000)
            self.height, self.width = self.mv_iterator.get_size()
            
            # Initialize frame generator for embedded preview
            self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
                sensor_width=self.width, sensor_height=self.height, 
                fps=self.event_fps_limit, palette=ColorPalette.Dark
            )
            
            self._notify_status(f"Event camera connected successfully: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            self._notify_status(f"Event camera connection failed: {e}")
            try:
                if self.device:
                    self.device = None
                self.mv_iterator = None
                self.event_frame_gen = None
            except:
                pass
            return False
    
    def disconnect_camera(self):
        """Disconnect camera gracefully"""
        self.stop_all()
        self.device = None
        self.mv_iterator = None
        self.event_frame_gen = None
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
        if not self.capturing:
            if not self.start_capture():
                return False
        
        if not self.device or self.visualization_running or not self.event_frame_gen:
            return False
        
        self.viz_thread = threading.Thread(target=self._visualization_thread, daemon=True)
        self.viz_thread.start()
        
        # Start separate preview thread for better performance
        self.preview_thread = threading.Thread(target=self._preview_worker, daemon=True)
        self.preview_thread.start()
        
        return True
    
    def stop_visualization(self):
        """Stop visualization"""
        if self.visualization_running:
            self.visualization_running = False
            
            if self.viz_thread:
                self.viz_thread.join(timeout=3.0)
            
            if self.preview_thread:
                self.preview_thread.join(timeout=3.0)
            
            # Clear the preview widget
            if self.preview_widget:
                self.preview_widget.configure(image='', text="Event Camera\nNot Active")
    
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
        """Thread for reading events from camera"""
        try:
            for evs in self.mv_iterator:
                if self.should_exit:
                    break
                
                try:
                    # Clear old events if queue is getting full
                    if self.event_queue.full():
                        try:
                            self.event_queue.get_nowait()
                        except:
                            pass
                    self.event_queue.put_nowait(evs)
                except queue.Full:
                    pass
                
                if self.should_exit:
                    break
        except Exception as e:
            self._notify_status(f"Event processing error: {e}")
        finally:
            self.capturing = False
    
    def _visualization_thread(self):
        """Thread for event frame generation - separate from preview"""
        try:
            frame_counter = 0
            
            def on_cd_frame_cb(ts, cd_frame):
                nonlocal frame_counter
                frame_counter += 1
                
                # Add status overlay less frequently for performance
                if frame_counter % 10 == 0:  # Only every 10th frame
                    overlay_frame = cd_frame.copy()
                    if self.is_recording:
                        cv2.putText(overlay_frame, "Event: RECORDING", (10, 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    else:
                        cv2.putText(overlay_frame, "Event: Live", (10, 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                else:
                    overlay_frame = cd_frame
                
                # Send to preview queue with frame dropping if full
                try:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()  # Drop old frame
                        except:
                            pass
                    self.frame_queue.put_nowait(overlay_frame)
                except queue.Full:
                    pass
            
            self.event_frame_gen.set_output_callback(on_cd_frame_cb)
            self.visualization_running = True
            self._notify_status("Event visualization started - embedded mode")
            
            # Process events with timing control
            event_counter = 0
            while not self.should_exit and self.visualization_running:
                try:
                    evs = self.event_queue.get(timeout=0.01)
                    event_counter += 1
                    
                    # Process every Nth event batch for performance
                    if event_counter % 2 == 0:  # Process every other event batch
                        self.event_frame_gen.process_events(evs)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    break
            
        except Exception as e:
            self._notify_status(f"Visualization error: {e}")
        finally:
            self.visualization_running = False
            self._notify_status("Event visualization stopped")
    
    def _preview_worker(self):
        """Separate thread for updating preview widget"""
        last_update_time = time.time()
        
        while not self.should_exit and self.visualization_running:
            try:
                cd_frame = self.frame_queue.get(timeout=0.05)
                
                current_time = time.time()
                
                # Limit preview update rate
                if current_time - last_update_time >= (1.0 / self.event_fps_limit):
                    if self.preview_widget and cd_frame is not None:
                        # Get current scale factor
                        scale_factor = 1.0
                        if self.scale_callback:
                            scale_factor = self.scale_callback()
                        
                        # Get widget dimensions
                        widget_width = self.preview_widget.winfo_width()
                        widget_height = self.preview_widget.winfo_height()
                        
                        if widget_width > 1 and widget_height > 1:
                            tk_image = cv2_to_tkinter_optimized(cd_frame, widget_width, widget_height, scale_factor)
                            if tk_image:
                                try:
                                    self.preview_widget.configure(image=tk_image, text="")
                                    self.preview_widget.image = tk_image
                                except:
                                    pass
                    
                    last_update_time = current_time
                        
            except queue.Empty:
                continue
            except Exception as e:
                break
        
        # Clear preview when stopped
        if self.preview_widget:
            try:
                self.preview_widget.configure(image='', text="Event Camera\nNot Active")
                self.preview_widget.image = None
            except:
                pass
    
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
    """Main GUI application with optimized embedded previews"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dual Camera Control System - Optimized Embedded Previews")
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Make window almost full screen
        main_width = screen_width - 20
        main_height = screen_height - 90
        self.root.geometry(f"{main_width}x{main_height}+0+0")
        
        # Preview scaling variables
        self.frame_scale_var = tk.DoubleVar(value=100.0)  # Default 100%
        self.event_scale_var = tk.DoubleVar(value=100.0)  # Default 100%
        self.frame_scale_text_var = tk.StringVar(value="100")
        self.event_scale_text_var = tk.StringVar(value="100")
        
        # Create preview widgets first
        self.frame_preview_widget = None
        self.event_preview_widget = None
        
        # Camera controllers
        self.frame_camera = None
        self.event_camera = None
        
        # GUI variables
        self.frame_cameras_list = []
        self.event_cameras_list = []
        
        # Status update queue
        self.status_queue = queue.Queue()
        
        # Parameter variables
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
    
    def get_frame_scale_factor(self):
        """Get current frame camera scale factor"""
        return self.frame_scale_var.get() / 100.0
    
    def get_event_scale_factor(self):
        """Get current event camera scale factor"""
        return self.event_scale_var.get() / 100.0
    
    def create_widgets(self):
        """Create GUI widgets with optimized embedded previews"""
        # Main horizontal paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Left side - controls (60%)
        controls_frame = ttk.Frame(main_paned)
        main_paned.add(controls_frame, weight=60)
        
        # Right side - previews (40%)
        preview_frame = ttk.Frame(main_paned)
        main_paned.add(preview_frame, weight=40)
        
        # Create controls in left frame
        self.create_controls_section(controls_frame)
        
        # Create preview section in right frame
        self.create_preview_section(preview_frame)
        
        # Initialize camera controllers with preview widgets and scale callbacks
        self.frame_camera = FrameCameraController(
            self.update_status, 
            self.frame_preview_widget, 
            self.get_frame_scale_factor
        )
        self.event_camera = EventCameraController(
            self.update_status, 
            self.event_preview_widget, 
            self.get_event_scale_factor
        )
    
    def create_controls_section(self, parent):
        """Create all control widgets"""
        # Main container with scrollable frame
        main_canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=main_canvas.yview)
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
        
        # Status section
        status_section = ttk.LabelFrame(scrollable_frame, text="Status Log")
        status_section.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.create_status_section(status_section)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_preview_section(self, parent):
        """Create optimized embedded preview section"""
        # Preview scaling controls
        scale_control_frame = ttk.LabelFrame(parent, text="Preview Scaling (50% - 200%)")
        scale_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Frame camera scale
        ttk.Label(scale_control_frame, text="Frame Scale:").grid(row=0, column=0, padx=5, pady=2)
        frame_scale = ttk.Scale(scale_control_frame, from_=50, to=200, variable=self.frame_scale_var,
                               orient=tk.HORIZONTAL, command=self.on_frame_scale_change)
        frame_scale.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        frame_scale_entry = ttk.Entry(scale_control_frame, textvariable=self.frame_scale_text_var, width=8)
        frame_scale_entry.grid(row=0, column=2, padx=5, pady=2)
        frame_scale_entry.bind('<Return>', self.on_frame_scale_entry_change)
        frame_scale_entry.bind('<FocusOut>', self.on_frame_scale_entry_change)
        
        ttk.Label(scale_control_frame, text="%").grid(row=0, column=3, padx=2, pady=2)
        
        # Event camera scale
        ttk.Label(scale_control_frame, text="Event Scale:").grid(row=1, column=0, padx=5, pady=2)
        event_scale = ttk.Scale(scale_control_frame, from_=50, to=200, variable=self.event_scale_var,
                               orient=tk.HORIZONTAL, command=self.on_event_scale_change)
        event_scale.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        
        event_scale_entry = ttk.Entry(scale_control_frame, textvariable=self.event_scale_text_var, width=8)
        event_scale_entry.grid(row=1, column=2, padx=5, pady=2)
        event_scale_entry.bind('<Return>', self.on_event_scale_entry_change)
        event_scale_entry.bind('<FocusOut>', self.on_event_scale_entry_change)
        
        ttk.Label(scale_control_frame, text="%").grid(row=1, column=3, padx=2, pady=2)
        
        scale_control_frame.columnconfigure(1, weight=1)
        
        # Vertical paned window for top/bottom previews
        preview_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        preview_paned.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Top preview - Frame Camera
        frame_preview_frame = ttk.LabelFrame(preview_paned, text="Frame Camera Preview")
        preview_paned.add(frame_preview_frame, weight=1)
        
        self.frame_preview_widget = ttk.Label(
            frame_preview_frame, 
            text="Frame Camera\nNot Active", 
            anchor=tk.CENTER,
            background="black",
            foreground="white",
            font=('TkDefaultFont', 12)
        )
        self.frame_preview_widget.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Bottom preview - Event Camera
        event_preview_frame = ttk.LabelFrame(preview_paned, text="Event Camera Preview")
        preview_paned.add(event_preview_frame, weight=1)
        
        self.event_preview_widget = ttk.Label(
            event_preview_frame, 
            text="Event Camera\nNot Active", 
            anchor=tk.CENTER,
            background="black",
            foreground="white",
            font=('TkDefaultFont', 12)
        )
        self.event_preview_widget.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    def on_frame_scale_change(self, value):
        """Handle frame scale change"""
        scale_value = float(value)
        self.frame_scale_text_var.set(f"{scale_value:.0f}")
    
    def on_frame_scale_entry_change(self, event=None):
        """Handle frame scale text entry change"""
        try:
            scale_value = float(self.frame_scale_text_var.get())
            scale_value = max(50, min(200, scale_value))  # Clamp to valid range
            self.frame_scale_var.set(scale_value)
            self.frame_scale_text_var.set(f"{scale_value:.0f}")
        except ValueError:
            self.frame_scale_text_var.set(f"{self.frame_scale_var.get():.0f}")
    
    def on_event_scale_change(self, value):
        """Handle event scale change"""
        scale_value = float(value)
        self.event_scale_text_var.set(f"{scale_value:.0f}")
    
    def on_event_scale_entry_change(self, event=None):
        """Handle event scale text entry change"""
        try:
            scale_value = float(self.event_scale_text_var.get())
            scale_value = max(50, min(200, scale_value))  # Clamp to valid range
            self.event_scale_var.set(scale_value)
            self.event_scale_text_var.set(f"{scale_value:.0f}")
        except ValueError:
            self.event_scale_text_var.set(f"{self.event_scale_var.get():.0f}")
    
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
        
        # Bias parameters frame (simplified for space)
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
    
    def create_status_section(self, parent):
        """Create status section"""
        self.status_text = tk.Text(parent, height=6, state=tk.DISABLED)  # Reduced height
        status_scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scroll.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    # [All the remaining methods stay the same as before - scan_cameras, connect methods, toggles, etc.]
    def scan_cameras(self):
        """Scan for available cameras"""
        if self.frame_camera:
            self.frame_cameras_list = self.frame_camera.find_cameras()
            frame_names = [f"{idx}: {name}" for idx, name, _ in self.frame_cameras_list]
            self.frame_camera_combo['values'] = frame_names
            if frame_names:
                self.frame_camera_combo.current(0)
        
        if self.event_camera:
            self.event_cameras_list = self.event_camera.find_cameras()
            event_names = [f"{idx}: {name}" for idx, name, _ in self.event_cameras_list]
            self.event_camera_combo['values'] = event_names
            if event_names:
                self.event_camera_combo.current(0)
        
        frame_count = len(self.frame_cameras_list) if hasattr(self, 'frame_cameras_list') else 0
        event_count = len(self.event_cameras_list) if hasattr(self, 'event_cameras_list') else 0
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
        """Connect to event camera"""
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
        """Toggle frame preview"""
        if not self.frame_camera.show_preview:
            if self.frame_camera.start_preview():
                self.frame_preview_btn.config(text="Stop Preview")
                if self.frame_camera.is_grabbing:
                    self.frame_grab_btn.config(text="Stop Grabbing")
        else:
            self.frame_camera.stop_preview()
            self.frame_preview_btn.config(text="Start Preview")
    
    def toggle_frame_recording(self):
        """Toggle frame recording"""
        if not self.frame_camera.is_recording:
            prefix = self.filename_prefix_var.get()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_frame_{timestamp}.avi" if prefix else None
            
            if self.frame_camera.start_recording(filename):
                self.frame_record_btn.config(text="Stop Recording")
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
        """Toggle event visualization"""
        if not self.event_camera.visualization_running:
            if self.event_camera.start_visualization():
                self.event_viz_btn.config(text="Stop Visualization")
                if self.event_camera.capturing:
                    self.event_capture_btn.config(text="Stop Capture")
        else:
            self.event_camera.stop_visualization()
            self.event_viz_btn.config(text="Start Visualization")
    
    def toggle_event_recording(self):
        """Toggle event recording"""
        if not self.event_camera.is_recording:
            prefix = self.filename_prefix_var.get()
            if self.event_camera.start_recording(filename_prefix=prefix):
                self.event_record_btn.config(text="Stop Recording")
                if self.event_camera.capturing:
                    self.event_capture_btn.config(text="Stop Capture")
        else:
            self.event_camera.stop_recording()
            self.event_record_btn.config(text="Start Recording")
    
    def toggle_unified_preview(self):
        """Toggle unified preview for both cameras"""
        if not self.unified_preview_active:
            frame_started = False
            event_started = False
            
            if self.frame_camera.cam is not None:
                frame_started = self.frame_camera.start_preview()
                if frame_started:
                    self.frame_preview_btn.config(text="Stop Preview")
                    if self.frame_camera.is_grabbing:
                        self.frame_grab_btn.config(text="Stop Grabbing")
            
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
            frame_started = False
            event_started = False
            prefix = self.filename_prefix_var.get()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.frame_camera.cam is not None:
                filename = f"{prefix}_frame_{timestamp}.avi" if prefix else None
                frame_started = self.frame_camera.start_recording(filename)
                if frame_started:
                    self.frame_record_btn.config(text="Stop Recording")
                    if self.frame_camera.is_grabbing:
                        self.frame_grab_btn.config(text="Stop Grabbing")
            
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
            pass
    
    def on_exposure_entry_change(self, event=None):
        """Handle exposure text entry change"""
        try:
            exposure = float(self.exposure_text_var.get())
            exposure_range = self.frame_camera.get_exposure_range()
            if exposure_range:
                min_exp, max_exp, _ = exposure_range
                exposure = max(min_exp, min(max_exp, exposure))
                self.exposure_var.set(exposure)
                self.exposure_text_var.set(f"{exposure:.0f}")
                self.frame_camera.set_exposure(exposure)
        except ValueError:
            self.exposure_text_var.set(f"{self.exposure_var.get():.0f}")
    
    def on_gain_scale_change(self, value):
        """Handle gain scale change"""
        gain = float(value)
        self.gain_text_var.set(f"{gain:.1f}")
        if self.frame_camera.set_gain(gain):
            pass
    
    def on_gain_entry_change(self, event=None):
        """Handle gain text entry change"""
        try:
            gain = float(self.gain_text_var.get())
            gain_range = self.frame_camera.get_gain_range()
            if gain_range:
                min_gain, max_gain, _ = gain_range
                gain = max(min_gain, min(max_gain, gain))
                self.gain_var.set(gain)
                self.gain_text_var.set(f"{gain:.1f}")
                self.frame_camera.set_gain(gain)
        except ValueError:
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
            bias_value = max(0, min(255, bias_value))
            self.bias_vars[bias_name].set(bias_value)
            self.bias_text_vars[bias_name].set(str(bias_value))
            self.event_camera.set_bias_value(bias_name, bias_value)
        except ValueError:
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
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    full_message = f"[{timestamp}] {message}\n"
                    
                    self.status_text.config(state=tk.NORMAL)
                    self.status_text.insert(tk.END, full_message)
                    self.status_text.see(tk.END)
                    self.status_text.config(state=tk.DISABLED)
            except queue.Empty:
                pass
            
            self.root.after(100, update)
        
        update()
    
    def run(self):
        """Run the GUI application"""
        def on_closing():
            try:
                if self.frame_camera:
                    self.frame_camera.force_disconnect()
                if self.event_camera:
                    self.event_camera.stop_all()
                time.sleep(0.2)
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