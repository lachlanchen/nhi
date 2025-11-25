# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# -- coding: utf-8 --

import sys
import threading
import msvcrt

from ctypes import *

sys.path.append("../MvImport")
from MVFGControl_class import *

BUFFER_NUMBER = 3
TIMEOUT       = 1000
g_bExit       = False


# ch:取流线程 | en:Grabbing image data thread
def GrabbingThread(Strm = 0):
    stFrameInfo = MV_FG_BUFFER_INFO()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    ret = Strm.StartAcquisition()
    if MV_FG_SUCCESS != ret:
        print("Warning: MV_FG_StartAcquisition error, ret[0x%x]" % ret)
        return ret
    while True:
        ret = Strm.GetFrameBuffer(stFrameInfo, TIMEOUT)
        if MV_FG_SUCCESS != ret:
            print("Warning: MV_FG_GetFrameBuffer error, ret[0x%x]" % ret)
            continue
        else:
            print("FrameNumber:[%d],    Width[%d],    Height[%d]" % (stFrameInfo.nFrameID, stFrameInfo.nWidth, stFrameInfo.nHeight))
        ret = Strm.ReleaseFrameBuffer(stFrameInfo)
        if MV_FG_SUCCESS != ret:
            print("Warning: MV_FG_ReleaseFrameBuffer error, ret[0x%x]" % ret)
        if True == g_bExit:
            break
    ret = Strm.StopAcquisition()
    if MV_FG_SUCCESS != ret:
        print("Warning: MV_FG_StopAcquisition error, ret[0x%x]" % ret)
        return ret
    return MV_FG_SUCCESS


# ch:打印采集卡信息 | en:Print interface info
def PrintInterfaceInfo(nInterfaceNum = 0):
    for i in range(0, nInterfaceNum):
        stInterfaceInfo = MV_FG_INTERFACE_INFO()
        memset(byref(stInterfaceInfo), 0, sizeof(stInterfaceInfo))
        ret = FGSystem.GetInterfaceInfo(i, stInterfaceInfo)
        if MV_FG_SUCCESS != ret:
            print("Warning: MV_FG_GetInterfaceInfo error, ret[0x%x]" % ret)
            return False

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
            print("[CXP]No.%d Interface: \n\tDisplayName: %s\n\tInterfaceID: %s\n\tSerialNumber:%s\n" % (i,
                    chDisplayName,
                    chInterfaceID,
                    chSerialNumber))

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
            print("[GEV]No.%d Interface: \n\tDisplayName: %s\n\tInterfaceID: %s\n\tSerialNumber:%s\n" % (i,
                  chDisplayName,
                  chInterfaceID,
                  chSerialNumber))

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
            print("[CML]No.%d Interface: \n\tDisplayName: %s\n\tInterfaceID: %s\n\tSerialNumber:%s\n" % (i,
                    chDisplayName,
                    chInterfaceID,
                    chSerialNumber))
                    
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
            print("[XoF]No.%d Interface: \n\tDisplayName: %s\n\tInterfaceID: %s\n\tSerialNumber:%s\n" % (i,
                    chDisplayName,
                    chInterfaceID,
                    chSerialNumber))

    return True

# ch:打印设备信息 | en:Print device info
def PrintDeviceInfo(Iface = 0, nDeviceNum = 0):
    for i in range(0, nDeviceNum):
        stDeviceInfo = MV_FG_DEVICE_INFO()
        memset(byref(stDeviceInfo), 0, sizeof(stDeviceInfo))
        ret = Iface.GetDeviceInfo(i, stDeviceInfo)
        if MV_FG_SUCCESS != ret:
            print("Warning: MV_FG_GetDeviceInfo error, ret[0x%x]" % ret)
            return False
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
            print("[CXP]No.%d Device: \n\tUserDefinedName: %s\n\tModelName: %s\n\tSerialNumber: %s\n" % (i,
                    chUserDefinedName,
                    chModelName,
                    chSerialNumber))

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
            print("[GEV]No.%d Device: \n\tUserDefinedName: %s\n\tModelName: %s\n\tSerialNumber: %s\n" % (i,
                    chUserDefinedName,
                    chModelName,
                    chSerialNumber))

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
            print("[CML]No.%d Device: \n\tUserDefinedName: %s\n\tModelName: %s\n\tSerialNumber: %s\n" % (i,
                    chUserDefinedName,
                    chModelName,
                    chSerialNumber))
                    
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
            print("[XoF]No.%d Device: \n\tUserDefinedName: %s\n\tModelName: %s\n\tSerialNumber: %s\n" % (i,
                    chUserDefinedName,
                    chModelName,
                    chSerialNumber))
                    
    return True


if __name__ == '__main__':

    Iface = FGInterface()
    Dev   = FGDevice()
    Strm  = FGStream()
    NULL  = c_void_p(None).value;
    stFrameInfo = MV_FG_BUFFER_INFO()
    for i in range(0, 1):
        # ch: 枚举采集卡 | en:Enum interface
        bChanged = c_bool(False)
        ret = FGSystem.UpdateInterfaceList(MV_FG_CXP_INTERFACE | MV_FG_GEV_INTERFACE | MV_FG_CAMERALINK_INTERFACE | MV_FG_XoF_INTERFACE, bChanged)
        if MV_FG_SUCCESS != ret:
            print("Warning: UpdateInterfaceList error, ret[0x%x]" % ret)
            break

        # ch: 获取采集卡数量 | en:Get interface num
        nInterfaceNum = c_uint(0)
        ret = FGSystem.GetNumInterfaces(nInterfaceNum)
        if MV_FG_SUCCESS != ret or 0 == nInterfaceNum.value:
            print("Warning: GetNumInterfaces error, ret[0x%x]" % ret)
            break

        # ch: 显示采集卡信息 | en:Show interfaceinfo
        ret = PrintInterfaceInfo(nInterfaceNum.value)
        if False == ret:
            break

        # ch: 选择采集卡 | en:Select interface
        nInterfaceIndex = input("Select an interface: ")
        if int(nInterfaceIndex) < 0 or int(nInterfaceIndex) >= nInterfaceNum.value:
            print("Invalid interface index.\nQuit.\n")
            break

        # ch: 打开采集卡，获得采集卡句柄 | en: Open interface, get handle
        ret = Iface.OpenInterface(int(nInterfaceIndex))
        if MV_FG_SUCCESS != ret:
            print("Warning: OpenInterface error, ret[0x%x]" % ret)
            break

        # ch: 枚举采集卡上的相机 | en:Enum camera of interface
        ret = Iface.UpdateDeviceList(bChanged)
        if MV_FG_SUCCESS != ret:
            print("Warning: UpdateDeviceList error, ret[0x%x]" % ret)
            break

        # ch: 获取设备数量 | en:Get device number
        nDeviceNum = c_uint(0)
        ret = Iface.GetNumDevices(nDeviceNum)
        if MV_FG_SUCCESS != ret:
            print("Warning: GetNumDevices error, ret[0x%x]" % ret)
            break

        # ch: 显示设备信息 | en:Show device info
        ret = PrintDeviceInfo(Iface, nDeviceNum.value)
        if False == ret or 0 == nDeviceNum.value:
            break

        # ch: 选择设备 | en:Select device
        nDeviceIndex = input("Select a device: ")
        if int(nDeviceIndex) < 0 or int(nDeviceIndex) >= nDeviceNum.value:
            print("Invalid device index.\nQuit.\n")
            break
        # ch: 打开设备，获得设备句柄 | en: Open device, get handle
        ret = Dev.OpenDevice(Iface, int(nDeviceIndex))
        if MV_FG_SUCCESS != ret:
            print("Warning: OpenDevice error, ret[0x%x]" % ret)
            break

        DevGeneral = FGGeneral(Dev)
        # ch:关闭触发模式 | en:Close trigger mode
        ret = DevGeneral.SetEnumValueByString("TriggerMode", "Off")
        if MV_FG_SUCCESS != ret:
            print("Warning: Set TriggerMode Off error, ret[0x%x]" % ret)
            break

        # ch:获取流通道个数 | en:Get number of stream
        nStreamNum = c_uint(0)
        ret = Dev.GetNumStreams(nStreamNum)
        if MV_FG_SUCCESS != ret or 0 == nStreamNum.value:
            print("Warning: GetNumStreams error, ret[0x%x]" % ret)
            break

        # ch:打开流通道(目前只支持单个通道) | en:Open stream(Only a single stream is supported now)
        ret = Strm.OpenStream(Dev, 0)
        if MV_FG_SUCCESS != ret:
            print("Warning: OpenStream error, ret[0x%x]" % ret)
            break

        # ch:设置SDK内部缓存数量 | en:Set internal buffer number
        ret = Strm.SetBufferNum(BUFFER_NUMBER)
        if MV_FG_SUCCESS != ret:
            print("Warning: SetBufferNum error, ret[0x%x]" % ret)
            break

        # ch:创建取流线程 | en:Create acquistion thread
        try:
            hThreadHandle = threading.Thread(target = GrabbingThread, args = (Strm,))
            hThreadHandle.start()
        except:
            print("error: unable to start thread")

        # ch:关闭取流线程 | en:Close acquistion thread
        print("press a key to stop grabbing.")
        msvcrt.getch()
        g_bExit = True
        hThreadHandle.join()

    # ch:关闭流通道 | en:Close Stream
    if Strm._STREAMHANDLE.value is not NULL:
        ret = Strm.CloseStream()
        if MV_FG_SUCCESS != ret:
            print("Warning: CloseStream error, ret[0x%x]" % ret)
        Strm.STREAMHANDLE = None

    # ch:关闭设备 | en:Close device
    if Dev._DEVHANDLE.value is not NULL:
        ret = Dev.CloseDevice()
        if MV_FG_SUCCESS != ret:
            print("Warning: CloseDevice error, ret[0x%x]" % ret)
        Dev.DEVHANDLE = None

    # ch:关闭采集卡 | en:Close interface
    if Iface._IFHANDLE.value is not NULL:
        ret = Iface.CloseInterface()
        if MV_FG_SUCCESS != ret:
            print("Warning: CloseInterface error, ret[0x%x]" % ret)
        Iface.IFHANDLE = None

    print("Press any key to exit.\n")
    msvcrt.getch()
    sys.exit()




















