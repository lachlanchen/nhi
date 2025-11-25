# [注] 需要保存文件的示例程序在部分环境下需以管理员权限执行，否则会有异常
# [PS] Sample programs that need to save files need to be executed with administrator privileges \
#      in some environments, otherwise there will be exceptions
     

import sys
import threading
import msvcrt
import numpy as np
import inspect
import csv
import codecs

from ctypes import *

sys.path.append("../MvImport")
from MVFGControl_class import *
from PyQt5.QtWidgets import *
from PyUIBasicDemo import Ui_Form


hThreadHandle = None
nFrameNum = 0                          #存图id
Buf_Lock = threading.Lock()            #取图和存图的锁
stImageInfo = MV_FG_INPUT_IMAGE_INFO()
Save_Image_Buf = None
Save_Image_Buf_Size = c_uint(0)

TIMEOUT         = 1000
nInterfaceNum   = c_uint(0)
IsOpenIF        = False
IsOpenDevice    = False
IsStartGrabbing = False
nTriggerMode    = c_uint(0)
TRIGGER_MODE_ON = c_uint(1)
TRIGGER_MODE_OFF = c_uint(0)

TRIGGER_SOURCE_LINE0 = c_uint(0)                     # ch:Line0 | en:Line0
TRIGGER_SOURCE_LINE1 = c_uint(1)                     # ch:Line1 | en:Line1
TRIGGER_SOURCE_LINE2 = c_uint(2)                     # ch:Line2 | en:Line2
TRIGGER_SOURCE_LINE3 = c_uint(3)                     # ch:Line3 | en:Line3
TRIGGER_SOURCE_COUNTER0 = c_uint(4)                  # ch:Conuter0 | en:Conuter0
TRIGGER_SOURCE_SOFTWARE = c_uint(7)                  # ch:软触发 | en:Software
TRIGGER_SOURCE_FrequencyConverter = c_uint(8)        # ch:变频器 | en:Frequency Converter

# 将返回的错误码转换为十六进制显示
def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr

# 强制关闭线程
def Async_raise(tid, exctype):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

# 停止线程
def Stop_thread(thread):
    Async_raise(thread.ident, SystemExit)

def EnabelControls(IsCameraReady):
    ui.BtnEnumInterface.setEnabled(not IsOpenIF)
    ui.BtnOpenInterface.setEnabled(not IsOpenIF and nInterfaceNum.value > 0)
    ui.BtnCloseInterface.setEnabled(IsOpenIF and not IsOpenDevice)
    ui.ComboInterface.setEnabled(not IsOpenIF and nInterfaceNum.value > 0)

    ui.BtnEnumDevice.setEnabled(not IsOpenDevice and IsOpenIF)
    ui.BtnOpenDevice.setEnabled(not IsOpenDevice and IsCameraReady)
    ui.BtnCloseDevice.setEnabled(IsOpenDevice and IsCameraReady)
    ui.ComboDevice.setEnabled(not IsOpenDevice and IsCameraReady)

    ui.BtnStart.setEnabled(not IsStartGrabbing and IsOpenDevice and IsCameraReady)
    ui.BtnStop.setEnabled(IsStartGrabbing)

    ui.RadioContinuousMode.setEnabled(IsOpenDevice)
    ui.RadioTriggerMode.setEnabled(IsOpenDevice)
    ui.CheckTriggerbySoftware.setEnabled(IsOpenDevice)
    ui.BtnTriggerOnce.setEnabled(IsStartGrabbing and ui.CheckTriggerbySoftware.isChecked() and ui.RadioTriggerMode.isChecked())

    ui.BtnSaveBMP.setEnabled(IsStartGrabbing)
    ui.BtnSaveJPEG.setEnabled(IsStartGrabbing)
    ui.BtnSaveTIFF.setEnabled(IsStartGrabbing)
    ui.BtnSavePNG.setEnabled(IsStartGrabbing)


# ch:取流线程 | en:Grabbing image data thread
def GrabbingThread(Stream = 0, winHandle = 0):
    global hThreadHandle
    global Save_Image_Buf
    global Save_Image_Buf_Size
    stFrameInfo   = MV_FG_BUFFER_INFO()
    stDisplayInfo = MV_FG_DISPLAY_FRAME_INFO()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    ret = MV_FG_SUCCESS
    while True:
        ret = Stream.GetFrameBuffer(stFrameInfo, TIMEOUT)
        if MV_FG_SUCCESS != ret :
            if nTriggerMode is TRIGGER_MODE_OFF :
                strError = "Get Frame Buffer Failed! ret:" + ToHexStr(ret)
                print(strError)
            continue
        else:
            Buf_Lock.acquire()
            memset(byref(stImageInfo), 0, sizeof(stImageInfo))
            if Save_Image_Buf is None or stFrameInfo.nFilledSize > Save_Image_Buf_Size:
                if Save_Image_Buf is not None:
                    del Save_Image_Buf
                    Save_Image_Buf = None
                Save_Image_Buf = (c_ubyte * stFrameInfo.nFilledSize)()
                Save_Image_Buf_Size = stFrameInfo.nFilledSize
            memset(byref(Save_Image_Buf), 0, Save_Image_Buf_Size)
            cdll.msvcrt.memcpy(byref(Save_Image_Buf), cast(stFrameInfo.pBuffer, POINTER(c_ubyte * stFrameInfo.nFilledSize)), stFrameInfo.nFilledSize)
            stImageInfo.nWidth = stFrameInfo.nWidth
            stImageInfo.nHeight = stFrameInfo.nHeight
            stImageInfo.enPixelType = stFrameInfo.enPixelType
            stImageInfo.pImageBuf = cast(stFrameInfo.pBuffer, POINTER(c_ubyte))
            stImageInfo.nImageBufLen = stFrameInfo.nFilledSize
            global  nFrameNum
            nFrameNum = stFrameInfo.nFrameID
            Buf_Lock.release()

            #显示图像
            memset(byref(stDisplayInfo), 0, sizeof(stDisplayInfo))
            stDisplayInfo.nWidth = stFrameInfo.nWidth
            stDisplayInfo.nHeight = stFrameInfo.nHeight
            stDisplayInfo.enPixelType = stFrameInfo.enPixelType
            stDisplayInfo.pImageBuf = cast(stFrameInfo.pBuffer, POINTER(c_ubyte))
            stDisplayInfo.nImageBufLen = stFrameInfo.nFilledSize
            ret = ImgProc.DisplayOneFrame(winHandle, stDisplayInfo)
            if MV_FG_SUCCESS != ret:
                Stream.ReleaseFrameBuffer(stFrameInfo)
                strError = "Display One Frame Failed! ret:" + ToHexStr(ret)
                QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
                break

            ret = Stream.ReleaseFrameBuffer(stFrameInfo)
            if MV_FG_SUCCESS != ret:
                strError = "Release Frame Buffer Failed! ret:" + ToHexStr(ret)
                QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
                break

        if False == IsStartGrabbing:
            break

    return ret

# ch:枚举采集卡 | en:Enum interface
def EnumInterface():
    bChanged = c_bool(False)
    ret = FGSystem.UpdateInterfaceList(MV_FG_CXP_INTERFACE | MV_FG_GEV_INTERFACE | MV_FG_CAMERALINK_INTERFACE | MV_FG_XoF_INTERFACE, bChanged)
    if MV_FG_SUCCESS != ret:
        strError = "Enum Interfaces Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return

    ret = FGSystem.GetNumInterfaces(nInterfaceNum)
    if MV_FG_SUCCESS != ret:
        strError = "Get Num Interfaces Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return

    if 0 == nInterfaceNum.value:
        strError = "No Interface!"
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return

    if True == bChanged.value:
        ui.ComboInterface.clear()
        for i in range(0, nInterfaceNum.value):
            stInterfaceInfo = MV_FG_INTERFACE_INFO()
            memset(byref(stInterfaceInfo), 0, sizeof(stInterfaceInfo))
            ret = FGSystem.GetInterfaceInfo(i, stInterfaceInfo)
            if MV_FG_SUCCESS != ret:
                strError = "Get Interface Info Failed! ret:" + ToHexStr(ret)
                QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
                return

            if MV_FG_CXP_INTERFACE == stInterfaceInfo.nTLayerType:
                chDisplayName = ""
                for per in stInterfaceInfo.IfaceInfo.stCXPIfaceInfo.chDisplayName:
                    chDisplayName = chDisplayName + chr(per)
                chInterfaceID = ""
                for per in stInterfaceInfo.IfaceInfo.stCXPIfaceInfo.chInterfaceID:
                    chInterfaceID = chInterfaceID + chr(per)
                chSerialNumber = ""
                for per in stInterfaceInfo.IfaceInfo.stCXPIfaceInfo.chSerialNumber:
                    chSerialNumber = chSerialNumber + chr(per)
                strIFInfo = "CXP[" + str(i) + "]" + chDisplayName + "|" + chInterfaceID + "|" + chSerialNumber

            elif MV_FG_GEV_INTERFACE == stInterfaceInfo.nTLayerType:
                chDisplayName = ""
                for per in stInterfaceInfo.IfaceInfo.stGEVIfaceInfo.chDisplayName:
                    chDisplayName = chDisplayName + chr(per)
                chInterfaceID = ""
                for per in stInterfaceInfo.IfaceInfo.stGEVIfaceInfo.chInterfaceID:
                    chInterfaceID = chInterfaceID + chr(per)
                chSerialNumber = ""
                for per in stInterfaceInfo.IfaceInfo.stGEVIfaceInfo.chSerialNumber:
                    chSerialNumber = chSerialNumber + chr(per)
                strIFInfo = "GEV[" + str(i) + "]" + chDisplayName + "|" + chInterfaceID + "|" + chSerialNumber

            elif MV_FG_CAMERALINK_INTERFACE == stInterfaceInfo.nTLayerType:
                chDisplayName = ""
                for per in stInterfaceInfo.IfaceInfo.stCMLIfaceInfo.chDisplayName:
                    chDisplayName = chDisplayName + chr(per)
                chInterfaceID = ""
                for per in stInterfaceInfo.IfaceInfo.stCMLIfaceInfo.chInterfaceID:
                    chInterfaceID = chInterfaceID + chr(per)
                chSerialNumber = ""
                for per in stInterfaceInfo.IfaceInfo.stCMLIfaceInfo.chSerialNumber:
                    chSerialNumber = chSerialNumber + chr(per)
                strIFInfo = "CML[" + str(i) + "]" + chDisplayName + "|" + chInterfaceID + "|" + chSerialNumber
                
            elif MV_FG_XoF_INTERFACE == stInterfaceInfo.nTLayerType:
                chDisplayName = ""
                for per in stInterfaceInfo.IfaceInfo.stXoFIfaceInfo.chDisplayName:
                    chDisplayName = chDisplayName + chr(per)
                chInterfaceID = ""
                for per in stInterfaceInfo.IfaceInfo.stXoFIfaceInfo.chInterfaceID:
                    chInterfaceID = chInterfaceID + chr(per)
                chSerialNumber = ""
                for per in stInterfaceInfo.IfaceInfo.stXoFIfaceInfo.chSerialNumber:
                    chSerialNumber = chSerialNumber + chr(per)
                strIFInfo = "XoF[" + str(i) + "]" + chDisplayName + "|" + chInterfaceID + "|" + chSerialNumber

            ui.ComboInterface.addItem(strIFInfo)

    if nInterfaceNum.value > 0:
        ui.ComboInterface.setCurrentIndex(0)

    EnabelControls(False)


# ch:打开采集卡 | en:Open Interface
def OpenInterface():
    global IsOpenIF
    nInterfaceIndex = ui.ComboInterface.currentIndex()
    if nInterfaceIndex < 0 or nInterfaceIndex >= ui.ComboInterface.count():
        strError = "Please select valid index!"
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    ret = Iface.OpenInterface(nInterfaceIndex)
    if MV_FG_SUCCESS != ret:
        strError = "Open Interface Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    IsOpenIF = True
    EnabelControls(False)

# ch；关闭采集卡 | en:Close interface
def CloseInterface():
    global IsOpenIF
    if True == IsOpenDevice:
        CloseDevice()
    ret = Iface.CloseInterface()
    if MV_FG_SUCCESS != ret:
        strError = "Close Interface Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    IsOpenIF = False

    EnabelControls(False)

# ch:枚举设备 | en:Enum device
def EnumDevice():
    bChanged   = c_bool(False)
    nDeviceNum = c_uint(0)

    ret = Iface.UpdateDeviceList(bChanged)
    if MV_FG_SUCCESS != ret:
        strError = "Enum Devices Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    ret = Iface.GetNumDevices(nDeviceNum)
    if MV_FG_SUCCESS != ret:
        strError = "Get Num Devices Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    if 0 == nDeviceNum.value:
        strError = "No Device!"
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    if True == bChanged.value:
        ui.ComboDevice.clear()
        for i in range(0, nDeviceNum.value  ):
            stDeviceInfo = MV_FG_DEVICE_INFO()
            memset(byref(stDeviceInfo), 0, sizeof(stDeviceInfo))
            ret = Iface.GetDeviceInfo(i, stDeviceInfo)
            if MV_FG_SUCCESS != ret:
                strError = "Get Device Info Failed! ret:" + ToHexStr(ret)
                QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
                return

            if MV_FG_CXP_DEVICE == stDeviceInfo.nDevType:
                chUserDefinedName = ""
                for per in stDeviceInfo.DevInfo.stCXPDevInfo.chUserDefinedName:
                    chUserDefinedName = chUserDefinedName + chr(per)
                chModelName = ""
                for per in stDeviceInfo.DevInfo.stCXPDevInfo.chModelName:
                    chModelName = chModelName + chr(per)
                chSerialNumber = ""
                for per in stDeviceInfo.DevInfo.stCXPDevInfo.chSerialNumber:
                    chSerialNumber = chSerialNumber + chr(per)
                strDevInfo = "CXP[" + str(i) + "]" + chUserDefinedName + "|" + chModelName + "|" + chSerialNumber

            elif MV_FG_GEV_DEVICE == stDeviceInfo.nDevType:
                chUserDefinedName = ""
                for per in stDeviceInfo.DevInfo.stGEVDevInfo.chUserDefinedName:
                    chUserDefinedName = chUserDefinedName + chr(per)
                chModelName = ""
                for per in stDeviceInfo.DevInfo.stGEVDevInfo.chModelName:
                    chModelName = chModelName + chr(per)
                chSerialNumber = ""
                for per in stDeviceInfo.DevInfo.stGEVDevInfo.chSerialNumber:
                    chSerialNumber = chSerialNumber + chr(per)
                strDevInfo = "GEV[" + str(i) + "]" + chUserDefinedName + "|" + chModelName + "|" + chSerialNumber

            elif MV_FG_CAMERALINK_DEVICE == stDeviceInfo.nDevType:
                chUserDefinedName = ""
                for per in stDeviceInfo.DevInfo.stCMLDevInfo.chUserDefinedName:
                    chUserDefinedName = chUserDefinedName + chr(per)
                chModelName = ""
                for per in stDeviceInfo.DevInfo.stCMLDevInfo.chModelName:
                    chModelName = chModelName + chr(per)
                chSerialNumber = ""
                for per in stDeviceInfo.DevInfo.stCMLDevInfo.chSerialNumber:
                    chSerialNumber = chSerialNumber + chr(per)
                strDevInfo = "CML[" + str(i) + "]" + chUserDefinedName + "|" + chModelName + "|" + chSerialNumber
                
            elif MV_FG_XoF_DEVICE == stDeviceInfo.nDevType:
                chUserDefinedName = ""
                for per in stDeviceInfo.DevInfo.stXoFDevInfo.chUserDefinedName:
                    chUserDefinedName = chUserDefinedName + chr(per)
                chModelName = ""
                for per in stDeviceInfo.DevInfo.stXoFDevInfo.chModelName:
                    chModelName = chModelName + chr(per)
                chSerialNumber = ""
                for per in stDeviceInfo.DevInfo.stXoFDevInfo.chSerialNumber:
                    chSerialNumber = chSerialNumber + chr(per)
                strDevInfo = "XoF[" + str(i) + "]" + chUserDefinedName + "|" + chModelName + "|" + chSerialNumber

            ui.ComboDevice.addItem(strDevInfo)

    if nDeviceNum.value > 0:
        ui.ComboDevice.setCurrentIndex(0)

    EnabelControls(True)

# ch:打开设备 | en:Open device
def OpenDevice():
    global IsOpenDevice
    if True == IsOpenDevice:
        return
    nIndex = ui.ComboDevice.currentIndex()
    if nIndex < 0 or nIndex >= ui.ComboDevice.count():
        strError = "Please select valid index!"
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    ret = Dev.OpenDevice(Iface, nIndex)
    if MV_FG_SUCCESS != ret:
        strError = "Open Device Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    IsOpenDevice = True
    global DevGeneral
    DevGeneral = FGGeneral(Dev)
    ret = GetTriggerMode()
    if MV_FG_SUCCESS != ret:
        strError = "Get Trigger Mode Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    ret = GetTriggerSource()
    if MV_FG_SUCCESS != ret:
        strError = "Get Trigger Source Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return

    EnabelControls(True)

# ch:关闭设备 | en:Close device
def CloseDevice():
    global IsOpenDevice
    if True == IsStartGrabbing:
        StopGrabbing()
    if True == IsOpenDevice:
        ret = Dev.CloseDevice()
        if MV_FG_SUCCESS != ret:
            strError = "Close Device Failed! ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
    IsOpenDevice = False

    EnabelControls(True)


# ch:获取触发模式 | en:Get trigger mode
def GetTriggerMode():
    stEnumValue = MV_FG_ENUMVALUE()
    memset(byref(stEnumValue), 0, sizeof(stEnumValue))
    ret = DevGeneral.GetEnumValue("TriggerMode", stEnumValue)
    if MV_FG_SUCCESS != ret:
        return ret
    global nTriggerMode
    nTriggerMode = stEnumValue.nCurValue
    if TRIGGER_MODE_ON == nTriggerMode:
        StartTriggerMode()
    else:
        StartContinuousMode()

    return MV_FG_SUCCESS

# ch:获取触发源 | en:Get trigger source
def GetTriggerSource():
    stEnumValue = MV_FG_ENUMVALUE()
    memset(byref(stEnumValue), 0, sizeof(stEnumValue))
    ret = DevGeneral.GetEnumValue("TriggerSource", stEnumValue)
    if MV_FG_SUCCESS == ret:
        return ret
    if TRIGGER_SOURCE_SOFTWARE != stEnumValue.nCurValue:
       ui.CheckTriggerbySoftware.setChecked(True)
    else:
       ui.CheckTriggerbySoftware.setChecked(False)
    return ret

# ch:设置触发模式 | en:Set trigger mode
def SetTriggerMode(nTriggerMode = 0):
    return DevGeneral.SetEnumValue("TriggerMode", nTriggerMode)

# ch:设置触发源 | en:Set trigger source
def SetTriggerSource():
    if True == ui.CheckTriggerbySoftware.isChecked():
        ret = DevGeneral.SetEnumValue("TriggerSource", TRIGGER_SOURCE_SOFTWARE)
        if MV_FG_SUCCESS != ret:
            strError = "Set Software Trigger Failed! ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            return
    else:
        ret = DevGeneral.SetEnumValue("TriggerSource", TRIGGER_SOURCE_LINE0)
        if MV_FG_SUCCESS != ret:
            strError = "Set Hardware Trigger Failed! ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            return


# ch:开启连续模式 | en:Start continuous mode
def StartContinuousMode():
    ui.RadioTriggerMode.setChecked(False)
    ui.RadioContinuousMode.setChecked(True)
    global nTriggerMode
    nTriggerMode = TRIGGER_MODE_OFF
    ret = SetTriggerMode(nTriggerMode)
    if MV_FG_SUCCESS != ret:
        strError = "Set Continuous Mode Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return


# ch:开启触发模式 | en:Start trigger mode
def StartTriggerMode():
    ui.RadioTriggerMode.setChecked(True)
    ui.RadioContinuousMode.setChecked(False)
    global nTriggerMode
    nTriggerMode = TRIGGER_MODE_ON
    ret = SetTriggerMode(nTriggerMode)
    if MV_FG_SUCCESS != ret:
        strError = "Set Trigger Mode Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        ui.RadioTriggerMode.setAutoExclusive(False)
        ui.RadioTriggerMode.setChecked(False)
        return

# ch:软触发 | en:Software trigger
def SoftwareTrigger():
    if True != IsStartGrabbing:
        return
    ret = DevGeneral.SetCommandValue("TriggerSoftware")
    if MV_FG_SUCCESS != ret:
        strError = "Software Trigger Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)

    EnabelControls(True)


# ch:开始采集 | en:Start grabbing
def StartGrabbing():
    global IsStartGrabbing
    global hThreadHandle
    if False == IsOpenDevice or True == IsStartGrabbing:
        return

    nStreamNum = c_uint(0)
    ret = Dev.GetNumStreams(nStreamNum)
    if MV_FG_SUCCESS != ret:
        strError = "Get Num Streams Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return

    ret = Stream.OpenStream(Dev, c_uint(0))
    if MV_FG_SUCCESS != ret:
        strError = "Open Stream Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    global ImgProc
    ImgProc = FGImageProcess(Stream)
    nBufferNum = c_uint(3)
    ret = Stream.SetBufferNum(nBufferNum)
    if MV_FG_SUCCESS != ret:
        strError = "Set Buffer Num Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return

    ret = Stream.StartAcquisition()
    if MV_FG_SUCCESS != ret:
        strError = "Start Acquisition Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return

    IsStartGrabbing = True
    try:
        hThreadHandle = threading.Thread(target=GrabbingThread, args=(Stream, int(ui.label_3.winId())))
        hThreadHandle.start()
    except:
        strError = "Start Thread Failed!"
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return

    EnabelControls(True)

# ch:停止采集 | en:Stop grabbing
def StopGrabbing():
    global IsStartGrabbing
    global hThreadHandle
    if False == IsOpenDevice or False == IsStartGrabbing:
        return
    IsStartGrabbing = False
    #hThreadHandle.join()
    Stop_thread(hThreadHandle)

    ret = Stream.StopAcquisition()
    if MV_FG_SUCCESS != ret:
        strError = "Stop Acquisition Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return

    ret = Stream.CloseStream()
    if MV_FG_SUCCESS != ret:
        strError = "Close Stream Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return

    EnabelControls(True)


# ch:保存BMP图像 | en:Save BMP
def SaveBmp():

    if 0 == Save_Image_Buf:
        return

    Buf_Lock.acquire()
    for i in range(1):
        file_path = str(nFrameNum) + ".bmp"
        stBmpInfo = MV_FG_SAVE_BITMAP_INFO()
        memset(byref(stBmpInfo), 0, sizeof(MV_FG_SAVE_BITMAP_INFO))
        BmpBuffer = (c_ubyte * (Save_Image_Buf_Size * 4 + 1028))()
        BmpBufferSize = Save_Image_Buf_Size * 4 + 1028

        stBmpInfo.stInputImageInfo = stImageInfo
        stBmpInfo.pBmpBuf = BmpBuffer
        stBmpInfo.nBmpBufSize = BmpBufferSize
        stBmpInfo.enCfaMethod = MV_FG_CFA_METHOD_OPTIMAL

        ret = ImgProc.SaveBitmap(stBmpInfo)
        if MV_FG_SUCCESS != ret:
            break
        file = open(file_path.encode('ascii'), 'wb+')
        file.write(BmpBuffer)
        file.close()

    Buf_Lock.release()

    if MV_FG_SUCCESS != ret:
        strError = "Save Bmp Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    QMessageBox.warning(mainWindow, "PROMPT", "Save Bmp Succeed!", QMessageBox.Ok)

# ch:保存JPEG图像 | en:Save JPEG
def SaveJpeg():

    if 0 == Save_Image_Buf:
        return

    Buf_Lock.acquire()
    for i in range(1):
        file_path = str(nFrameNum) + ".jpg"
        stJpegInfo = MV_FG_SAVE_JPEG_INFO()
        memset(byref(stJpegInfo), 0, sizeof(MV_FG_SAVE_JPEG_INFO))
        JpegBuffer = (c_ubyte * (Save_Image_Buf_Size * 4 + 1028))()
        JpegBufferSize = Save_Image_Buf_Size * 4 + 1028

        stJpegInfo.stInputImageInfo = stImageInfo
        stJpegInfo.pJpgBuf = JpegBuffer
        stJpegInfo.nJpgBufSize = JpegBufferSize
        stJpegInfo.nJpgQuality = 60                              # JPG编码质量(0 - 100]
        stJpegInfo.enCfaMethod = MV_FG_CFA_METHOD_OPTIMAL

        ret = ImgProc.SaveJpeg(stJpegInfo)
        if MV_FG_SUCCESS != ret:
            break
        file = open(file_path.encode('ascii'), 'wb+')
        file.write(JpegBuffer)
        file.close()

    Buf_Lock.release()

    if MV_FG_SUCCESS != ret:
        strError = "Save Jpeg Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    QMessageBox.warning(mainWindow, "PROMPT", "Save Jpeg Succeed!", QMessageBox.Ok)

# ch:保存TIFF图像 | en:Save TIFF
def SaveTiff():
    if 0 == Save_Image_Buf:
        return

    Buf_Lock.acquire()
    file_path = str(nFrameNum) + ".tif"
    stTiffInfo = MV_FG_SAVE_TIFF_TO_FILE_INFO()
    memset(byref(stTiffInfo), 0, sizeof(MV_FG_SAVE_TIFF_TO_FILE_INFO))
    ImagePath = (c_ubyte * 256)()
    cdll.msvcrt.memcpy(byref(ImagePath), file_path.encode('ascii'), 256)

    stTiffInfo.stInputImageInfo = stImageInfo
    stTiffInfo.fXResolution = stImageInfo.nWidth
    stTiffInfo.fYResolution = stImageInfo.nHeight
    stTiffInfo.enResolutionUnit = MV_FG_Resolution_Unit_Inch
    stTiffInfo.enCfaMethod = MV_FG_CFA_METHOD_OPTIMAL
    stTiffInfo.pcImagePath = ImagePath

    ret = ImgProc.SaveTiffToFile(stTiffInfo)
    Buf_Lock.release()

    if MV_FG_SUCCESS != ret:
        strError = "Save Tiff Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    QMessageBox.warning(mainWindow, "PROMPT", "Save Tiff Succeed!", QMessageBox.Ok)

# ch:保存PNG图像 | en:Save PNG
def SavePng():

    if 0 == Save_Image_Buf:
        return

    Buf_Lock.acquire()
    file_path = str(nFrameNum) + ".png"
    stPngInfo = MV_FG_SAVE_PNG_TO_FILE_INFO()
    memset(byref(stPngInfo), 0, sizeof(MV_FG_SAVE_PNG_TO_FILE_INFO))
    ImagePath = (c_ubyte * 256)()
    cdll.msvcrt.memcpy(byref(ImagePath), file_path.encode('ascii'), 256)

    stPngInfo.stInputImageInfo = stImageInfo
    stPngInfo.nPngCompression = 6
    stPngInfo.enCfaMethod = MV_FG_CFA_METHOD_OPTIMAL
    stPngInfo.pcImagePath = ImagePath

    ret = ImgProc.SavePngToFile(stPngInfo)
    Buf_Lock.release()

    if MV_FG_SUCCESS != ret:
        strError = "Save Png Failed! ret:" + ToHexStr(ret)
        QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        return
    QMessageBox.warning(mainWindow, "PROMPT", "Save Png Succeed!", QMessageBox.Ok)


if __name__ == '__main__':
    Iface = FGInterface()
    Dev = FGDevice()
    Stream = FGStream()

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    mainWindow.setFixedSize(787, 632)
    ui = Ui_Form()
    ui.setupUi(mainWindow)
    EnabelControls(False)
    ui.BtnEnumInterface.clicked.connect(EnumInterface)
    ui.BtnOpenInterface.clicked.connect(OpenInterface)
    ui.BtnCloseInterface.clicked.connect(CloseInterface)
    ui.BtnEnumDevice.clicked.connect(EnumDevice)
    ui.BtnOpenDevice.clicked.connect(OpenDevice)
    ui.BtnCloseDevice.clicked.connect(CloseDevice)
    ui.RadioContinuousMode.clicked.connect(StartContinuousMode)
    ui.RadioTriggerMode.clicked.connect(StartTriggerMode)
    ui.CheckTriggerbySoftware.clicked.connect(SetTriggerSource)
    ui.BtnTriggerOnce.clicked.connect(SoftwareTrigger)
    ui.BtnStart.clicked.connect(StartGrabbing)
    ui.BtnStop.clicked.connect(StopGrabbing)
    ui.BtnSaveBMP.clicked.connect(SaveBmp)
    ui.BtnSaveJPEG.clicked.connect(SaveJpeg)
    ui.BtnSaveTIFF.clicked.connect(SaveTiff)
    ui.BtnSavePNG.clicked.connect(SavePng)

    mainWindow.show()
    app.exec_()

    if True == IsOpenDevice:
        CloseDevice()
    if True == IsOpenIF:
        CloseInterface()

    sys.exit()


