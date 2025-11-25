# -- coding: utf-8 --

import sys
import threading
import msvcrt

sys.path.append("../MvImport")
from MvCameraControl_class import *

g_bExit = False


# 为线程定义一个函数
def work_thread(cam=0, pData=0, nDataSize=0):
    out_frame_info = MV_FRAME_OUT()
    memset(byref(out_frame_info), 0, sizeof(out_frame_info))
    while True:
        ret = cam.MV_CC_GetImageBuffer(out_frame_info, 1000)
        if out_frame_info.pBufAddr is not None and 0 == ret:
            print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                out_frame_info.stFrameInfo.nWidth, out_frame_info.stFrameInfo.nHeight,
                out_frame_info.stFrameInfo.nFrameNum))
            ret = cam.MV_CC_FreeImageBuffer(out_frame_info)
        else:
            print("error: no data[0x%x]" % ret)
        if g_bExit is True:
            break


if __name__ == "__main__":

    # ch:初始化SDK | en: initialize SDK
    MvCamera.MV_CC_Initialize()

    device_list = MV_CC_DEVICE_INFO_LIST()
    t_layer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                    | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)

    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(t_layer_type, device_list)
    if ret != 0:
        print("error: enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if device_list.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print("Find %d devices!" % device_list.nDeviceNum)

    for i in range(0, device_list.nDeviceNum):
        mvcc_dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
            print("gige device: [%d]" % i)
            device_model_name = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                if per == 0:
                    break
                device_model_name = device_model_name + chr(per)
            print("device model name: %s" % device_model_name)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("u3v device: [%d]" % i)
            device_model_name = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                device_model_name = device_model_name + chr(per)
            print("device model name: %s" % device_model_name)

            serial_number = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                serial_number = serial_number + chr(per)
            print("user serial number: %s" % serial_number)
        elif mvcc_dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
            print ("\nCML device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stCMLInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print ("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stCMLInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print ("user serial number: %s" % strSerialNumber)
        elif mvcc_dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
            print ("\nCXP device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stCXPInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print ("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stCXPInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print ("user serial number: %s" % strSerialNumber)
        elif mvcc_dev_info.nTLayerType == MV_GENTL_XOF_DEVICE:
            print ("\nXoF device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stXoFInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print ("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stXoFInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print ("user serial number: %s" % strSerialNumber)

    connection_number = input("please input the number of the device to connect:")

    if int(connection_number) >= device_list.nDeviceNum:
        print("error: input error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    current_device_info = cast(device_list.pDeviceInfo[int(connection_number)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(current_device_info)
    if ret != 0:
        print("error: create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("error: open device fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if current_device_info.nTLayerType == MV_GIGE_DEVICE or current_device_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    b_enable = c_bool(False)
    ret = cam.MV_CC_GetBoolValue("AcquisitionLineRateEnable", b_enable)
    if ret != 0:
        print("error: get AcquisitionLineRateEnable fail! ret[0x%x]" % ret)

    # ch:设置行触发 | en:Set trigger selector is line start
    ret = cam.MV_CC_SetEnumValueByString("TriggerSelector", "LineStart")
    if ret != 0:
        print("error: set trigger selector fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:设置触发模式为On | en:Set trigger mode as on
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
    if ret != 0:
        print("error: set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:设置触发源 | en:Set trigger source
    ret = cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE1)
    if ret != 0:
        print("error: set trigger source fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("error: start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    try:
        thread_handle = threading.Thread(target=work_thread, args=(cam, None, None))
        thread_handle.start()
    except:
        print("error: unable to start thread")

    print("press a key to stop grabbing.")
    msvcrt.getch()

    g_bExit = True
    thread_handle.join()

    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("error: stop grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("error: close device fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("error: destroy handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:反初始化SDK | en: finalize SDK
    MvCamera.MV_CC_Finalize()
