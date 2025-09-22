# -- coding: utf-8 --
import threading
import time
import sys
import ctypes
from ctypes import *

sys.path.append("../MvImport")
from MvCameraControl_class import *

class CameraOperation():

    def __init__(self,obj_cam,st_device_list,n_connect_num=0,b_open_device=False,b_start_grabbing = False,h_thread_handle=None,\
                b_thread_opened=False,st_frame_info=None,b_save_bmp=False,b_save_jpg=False,buf_save_image=None):

        self.obj_cam = obj_cam
        self.st_device_list = st_device_list
        self.n_connect_num = n_connect_num
        self.b_open_device = b_open_device
        self.b_start_grabbing = b_start_grabbing 
        self.b_thread_opened = b_thread_opened
        self.st_frame_info = MV_FRAME_OUT_INFO_EX()
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.buf_save_image_len = 0
        self.h_thread_handle = h_thread_handle
        self.buf_lock = threading.Lock()  # 取图和存图的buffer锁
        self.exit_flag = 0
        self.frame_count = 0
        self.lost_frame_count = 0

    # 转为16进制字符串
    def to_hex_str(self, num):
        chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
        hexStr = ""
        if num < 0:
            num = num + 2**32
        while num >= 16:
            digit = num % 16
            hexStr = chaDic.get(digit, str(digit)) + hexStr
            num //= 16
        hexStr = chaDic.get(num, str(num)) + hexStr
        return hexStr

    # 打开相机
    def open_device(self):
        if self.b_open_device is False:
            # ch:选择设备并创建句柄 | en:Select device and create handle
            nConnectionNum = int(self.n_connect_num)
            stDeviceList = cast(self.st_device_list.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
            self.obj_cam = MvCamera()
            ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()
                return ret

            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                self.b_open_device = False
                self.b_thread_opened = False
                return ret
            self.b_open_device = True
            self.b_thread_opened = False

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                    if ret != 0:
                        print("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print("warning: packet size is invalid[%d]" % nPacketSize)

            stBool = c_bool(False)
            ret =self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print("warning: get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
            if ret != 0:
                print("warning: set trigger mode off fail! ret[0x%x]" % ret)
            return 0
        return 0
            
    # 开始取图
    def start_grabbing(self, n_index,  win_handle):
        if not self.b_start_grabbing and self.b_open_device:
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                self.b_start_grabbing = False
                return ret
            self.b_start_grabbing = True
            print("start grabbing " + str(n_index) + "successfully!")
            try:
                self.exit_flag = threading.Event()
                self.h_thread_handle = threading.Thread(target=CameraOperation.work_thread, args=(self, n_index, win_handle, self.exit_flag))
                self.h_thread_handle.start()
                self.b_thread_opened = True
            except TypeError:
                print('error: unable to start thread')
                self.b_start_grabbing = False
            return 0
        return MV_E_CALLORDER

    # 停止取图
    def stop_grabbing(self):
        if (self.b_start_grabbing is True) and (self.b_open_device is True):
            # 退出线程
            if self.b_thread_opened:
                self.exit_flag.set()
                self.h_thread_handle.join()
                self.b_thread_opened = False
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                return ret
            self.b_start_grabbing = False
            return 0
        return MV_E_CALLORDER

    # 关闭相机
    def close_device(self):
        if self.b_open_device:
            #退出线程
            if self.b_thread_opened:
                self.exit_flag.set()
                self.h_thread_handle.join()
                self.b_thread_opened = False
            if self.b_start_grabbing:
                ret = self.obj_cam.MV_CC_StopGrabbing()
                if ret != 0:
                    return ret
                self.b_start_grabbing = False
            ret = self.obj_cam.MV_CC_CloseDevice()
            if ret != 0:
                return ret
                
        # ch:销毁句柄 | Destroy handle
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        return 0

    # 设置触发模式
    def set_trigger_mode(self, trigger_mode):
        if True == self.b_open_device:
            if "continuous" == trigger_mode:
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode","Off")
                if ret != 0:
                    return ret
                return 0
            if "triggermode" == trigger_mode:
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerMode","On")
                if ret != 0:
                    return ret
                return 0

    # 设置触发源
    def set_trigger_source(self, trigger_source):
        if self.b_open_device is True:
            if "software" == trigger_source:
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerSource", "Software")
                if ret != 0:
                    return ret
                return 0
            else:
                ret = self.obj_cam.MV_CC_SetEnumValueByString("TriggerSource", "Line0")
                if ret != 0:
                    return ret
                return 0

    # 软触发一次
    def trigger_once(self):
        if self.b_open_device is True:
            ret = self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")
            return ret

    def set_exposure_time(self, str_value):
        if self.b_open_device is True:
            self.obj_cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            time.sleep(0.2)
            ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(str_value))
            if ret != 0:
                print('show error', 'set exposure time fail! ret = ' + self.to_hex_str(ret))
                return ret
        return 0

    def set_gain(self, str_value):
        if self.b_open_device is True:
            self.obj_cam.MV_CC_SetEnumValue("GainAuto", 0)
            time.sleep(0.2)
            ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(str_value))
            if ret != 0:
                print('show error', 'set gain fail! ret = ' + self.to_hex_str(ret))
                return ret
        return 0

    def set_frame_rate(self, str_value):
        if self.b_open_device is True:
            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(str_value))
            return ret
        return 0

    # 取图线程函数
    def work_thread(self, n_index, win_handle, exit_flag):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        while not exit_flag.is_set():
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if 0 == ret:
                # 拷贝图像和图像信息
                # 获取缓存锁
                self.buf_lock.acquire()
                if self.buf_save_image_len < stOutFrame.stFrameInfo.nFrameLen:
                    if self.buf_save_image is not None:
                        del self.buf_save_image
                        self.buf_save_image = None
                    self.buf_save_image = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                    self.buf_save_image_len = stOutFrame.stFrameInfo.nFrameLen

                cdll.msvcrt.memcpy(byref(self.st_frame_info), byref(stOutFrame.stFrameInfo), sizeof(MV_FRAME_OUT_INFO_EX))
                cdll.msvcrt.memcpy(byref(self.buf_save_image), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)
                self.buf_lock.release()

                # 使用Display接口显示图像
                stDisplayParam = MV_DISPLAY_FRAME_INFO()
                memset(byref(stDisplayParam), 0, sizeof(stDisplayParam))
                stDisplayParam.hWnd = int(win_handle)
                stDisplayParam.nWidth = stOutFrame.stFrameInfo.nWidth
                stDisplayParam.nHeight = stOutFrame.stFrameInfo.nHeight
                stDisplayParam.enPixelType = stOutFrame.stFrameInfo.enPixelType
                stDisplayParam.pData = stOutFrame.pBufAddr
                stDisplayParam.nDataLen = stOutFrame.stFrameInfo.nFrameLen
                self.obj_cam.MV_CC_DisplayOneFrame(stDisplayParam)

                # 释放缓存
                self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                print("Camera[" + str(n_index) + "]:no data, ret = "+self.to_hex_str(ret))
                continue

    # 存BMP图像
    def save_bmp(self):
        if 0 == self.buf_save_image:
            return

        # 获取缓存锁
        self.buf_lock.acquire()

        file_path = "cam" + str(self.n_connect_num) + "_" + str(self.st_frame_info.nFrameNum) + ".bmp"
        c_file_path = file_path.encode('ascii')

        stSaveParam = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
        stSaveParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stSaveParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stSaveParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stSaveParam.nDataLen = self.st_frame_info.nFrameLen
        stSaveParam.pData = cast(self.buf_save_image, POINTER(c_ubyte))
        stSaveParam.enImageType = MV_Image_Bmp  # ch:需要保存的图像类型 | en:Image format to save
        stSaveParam.pcImagePath = ctypes.create_string_buffer(c_file_path)
        stSaveParam.iMethodValue = 1
        ret = self.obj_cam.MV_CC_SaveImageToFileEx(stSaveParam)

        self.buf_lock.release()

        return ret
