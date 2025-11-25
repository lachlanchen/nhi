# -- coding: utf-8 --

import sys
import copy
import ctypes

from ctypes import *

from MVFGDefines_const import *
from MVFGDefines_header import *
from MVFGErrorDefine_const import *


# Python3.8版本修改Dll加载策略, 默认不再搜索Path环境变量, 同时增加winmode参数以兼容旧版本
dllname = "MVFGControl.dll"
if "winmode" in ctypes.WinDLL.__init__.__code__.co_varnames:
    MVFGCtrldll = WinDLL(dllname, winmode=0)
else:
    MVFGCtrldll = WinDLL(dllname)

# 用于回调函数传入相机实例
class _MV_PY_OBJECT_(Structure):
    pass
_MV_PY_OBJECT_._fields_ = [
    ('PyObject', py_object),
]
MV_PY_OBJECT = _MV_PY_OBJECT_

class FGSystem():

    @staticmethod
    def GetSDKVersion():
        MVFGCtrldll.MV_FG_GetSDKVersion.restype = c_char_p
        # C原型：unsigned char* __stdcall MV_FG_GetSDKVersion();
        return MVFGCtrldll.MV_FG_GetSDKVersion()

    # ch:更新采集卡列表 | en:Update Interface List
    @staticmethod
    def UpdateInterfaceList(nTLayerType, bChanged):
        MVFGCtrldll.MV_FG_UpdateInterfaceList.argtype = (c_uint, c_void_p)
        MVFGCtrldll.MV_FG_UpdateInterfaceList.restype = c_uint
        # C原型：int __stdcall MV_FG_UpdateInterfaceList(IN unsigned int nTLayerType, OUT bool8_t *pbChanged);
        return MVFGCtrldll.MV_FG_UpdateInterfaceList(nTLayerType, byref(bChanged))

    # ch:释放指定类型的采集卡资源 | en:Release Interface Resources of the Specified Type
    @staticmethod
    def ReleaseTLayerResource(nTLayerType):
        MVFGCtrldll.MV_FG_ReleaseTLayerResource.argtype = c_uint
        MVFGCtrldll.MV_FG_ReleaseTLayerResource.restype = c_uint
        # C原型：int __stdcall MV_FG_ReleaseTLayerResource(IN unsigned int nTLayerType);
        return MVFGCtrldll.MV_FG_ReleaseTLayerResource(nTLayerType)

    # ch:获取采集卡数量 | en:Get the Number of Interface
    @staticmethod
    def GetNumInterfaces(nNumIfaces):
        MVFGCtrldll.MV_FG_GetNumInterfaces.argtype = c_void_p
        MVFGCtrldll.MV_FG_GetNumInterfaces.restype = c_uint
        # C原型：int __stdcall MV_FG_GetNumInterfaces(OUT unsigned int *pnNumIfaces);
        return MVFGCtrldll.MV_FG_GetNumInterfaces(byref(nNumIfaces))

    # ch:根据索引获取采集卡信息 | Obtain Interface Information According to Index
    @staticmethod
    def GetInterfaceInfo(nIndex, stIfaceInfo):
        MVFGCtrldll.MV_FG_GetInterfaceInfo.argtype = (c_uint, c_void_p)
        MVFGCtrldll.MV_FG_GetInterfaceInfo.restype = c_uint
        # C原型：int __stdcall MV_FG_GetInterfaceInfo(IN unsigned int nIndex, OUT MV_FG_INTERFACE_INFO *pstIfaceInfo);
        return MVFGCtrldll.MV_FG_GetInterfaceInfo(nIndex, byref(stIfaceInfo))

class FGInterface():

    def __init__(self):
        self._IFHANDLE     = c_void_p()                      # 采集卡句柄
        self.IFHANDLE      = pointer(self._IFHANDLE)         # 创建句柄指针

    def GetHandle(self):
        return self.IFHANDLE

    # ch:打开采集卡 | en:Open Interface
    def OpenInterface(self, nIndex):
        MVFGCtrldll.MV_FG_OpenInterface.argtype = (c_uint, c_void_p)
        MVFGCtrldll.MV_FG_OpenInterface.restype = c_uint
        # C原型：int __stdcall MV_FG_OpenInterface(IN unsigned int nIndex, OUT IFHANDLE* phIface);
        return MVFGCtrldll.MV_FG_OpenInterface(nIndex, byref(self.IFHANDLE))

    # ch: 打开采集卡，可指定权限 | en:Open the Interface with Specify Permissions
    def OpenInterfaceEx(self, nIndex, nAccess):
        MVFGCtrldll.MV_FG_OpenInterfaceEx.argtype = (c_uint, c_int, c_void_p)
        MVFGCtrldll.MV_FG_OpenInterfaceEx.restype = c_uint
        # C原型：int __stdcall MV_FG_OpenInterfaceEx(IN unsigned int nIndex, IN int nAccess, OUT IFHANDLE* phIface);
        return MVFGCtrldll.MV_FG_OpenInterfaceEx(nIndex, nAccess, byref(self.IFHANDLE))

    # ch:通过采集卡ID打开采集卡，可指定权限 | en:Open the Interface with Specify Permissions
    def OpenInterfaceByID(self, strInterfaceID, nAccess):
        MVFGCtrldll.MV_FG_OpenInterfaceByID.argtype = (c_void_p, c_int, c_void_p)
        MVFGCtrldll.MV_FG_OpenInterfaceByID.restype = c_uint
        # C原型：int __stdcall MV_FG_OpenInterfaceByID(IN char* pcInterfaceID, IN int nAccess, OUT IFHANDLE* phIface);
        return MVFGCtrldll.MV_FG_OpenInterfaceByID(strInterfaceID.encode('ascii'), nAccess, byref(self.IFHANDLE))

    # ch: 打开采集卡，导入配置文件 | en:Open the Interface with config file
    def OpenInterfaceWithConfig(self, nIndex, strConfigFile):
        MVFGCtrldll.MV_FG_OpenInterfaceWithConfig.argtype = (c_uint, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_OpenInterfaceWithConfig.restype = c_uint
        # C原型：int __stdcall MV_FG_OpenInterfaceWithConfig(IN unsigned int nIndex, IN char* pcConfigFile, OUT IFHANDLE* phIface);
        return MVFGCtrldll.MV_FG_OpenInterfaceWithConfig(nIndex, strConfigFile.encode('ascii'), byref(self.IFHANDLE))

    # ch:关闭采集卡 | en:Close Interface
    def CloseInterface(self):
        MVFGCtrldll.MV_FG_CloseInterface.argtype = c_void_p
        MVFGCtrldll.MV_FG_CloseInterface.restype = c_uint
        # C原型：int __stdcall MV_FG_CloseInterface(IN IFHANDLE hIface);
        return MVFGCtrldll.MV_FG_CloseInterface(self.IFHANDLE)

    # ch:更新指定采集卡下的设备列表 | en:Update the Device List Under the Specified Interface
    def UpdateDeviceList(self, bChanged):
        MVFGCtrldll.MV_FG_UpdateDeviceList.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_UpdateDeviceList.restype = c_uint
        # C原型：int __stdcall MV_FG_UpdateDeviceList(IN IFHANDLE hIface, OUT bool8_t *pbChanged);
        return MVFGCtrldll.MV_FG_UpdateDeviceList(self.IFHANDLE, byref(bChanged))

    # ch:获取设备数量 | en:Get the Number of Devices
    def GetNumDevices(self, nNumDevices):
        MVFGCtrldll.MV_FG_GetNumDevices.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_GetNumDevices.restype = c_uint
        # C原型：int __stdcall MV_FG_GetNumDevices(IN IFHANDLE hIface, OUT unsigned int *pnNumDevices);
        return MVFGCtrldll.MV_FG_GetNumDevices(self.IFHANDLE, byref(nNumDevices))

    # ch:获取设备信息 | en:Get Device Information
    def GetDeviceInfo(self, Index, stDevInfo):
        MVFGCtrldll.MV_FG_GetDeviceInfo.argtype = (c_void_p, c_uint, c_void_p)
        MVFGCtrldll.MV_FG_GetDeviceInfo.restype = c_uint
        # C原型：int __stdcall MV_FG_GetDeviceInfo(IN IFHANDLE hIface, IN unsigned int nIndex, OUT MV_FG_DEVICE_INFO *pstDevInfo);
        return MVFGCtrldll.MV_FG_GetDeviceInfo(self.IFHANDLE, Index, byref(stDevInfo))


class FGDevice():
    def __init__(self):
        self._DEVHANDLE = c_void_p()  # 设备句柄
        self.DEVHANDLE = pointer(self._DEVHANDLE)  # 创建句柄指针

    def GetHandle(self):
        return self.DEVHANDLE

    # ch:打开设备 | en:Open Device
    def OpenDevice(self, hIface, nIndex):
        handle = hIface.GetHandle()
        MVFGCtrldll.MV_FG_OpenDevice.argtype = (c_void_p, c_uint, c_void_p)
        MVFGCtrldll.MV_FG_OpenDevice.restype = c_uint
        # C原型：int __stdcall MV_FG_OpenDevice(IN IFHANDLE hIface, IN unsigned int nIndex, OUT DEVHANDLE* phDevice);
        return MVFGCtrldll.MV_FG_OpenDevice(handle, nIndex, byref(self.DEVHANDLE))

    # ch:打开设备 | en:Open Device
    def OpenDeviceByID(self, hIface, strDeviceID):
        handle = hIface.GetHandle()
        MVFGCtrldll.MV_FG_OpenDeviceByID.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_OpenDeviceByID.restype = c_uint
        # C原型：int __stdcall MV_FG_OpenDeviceByID(IN IFHANDLE hIface, IN char* pcDeviceID, OUT DEVHANDLE* phDevice);
        return MVFGCtrldll.MV_FG_OpenDeviceByID(handle, strDeviceID.encode('ascii'), byref(self.DEVHANDLE))

    # ch:关闭设备 | en:Close Device
    def CloseDevice(self):
        MVFGCtrldll.MV_FG_CloseDevice.argtype = c_void_p
        MVFGCtrldll.MV_FG_CloseDevice.restype = c_uint
        # C原型：int __stdcall MV_FG_CloseDevice(IN DEVHANDLE hDevice);
        return MVFGCtrldll.MV_FG_CloseDevice(self.DEVHANDLE)

    # ch:获取流通道数量 | en:Get the Number of Streams
    def GetNumStreams(self, nNumStreams):
        MVFGCtrldll.MV_FG_GetNumStreams.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_GetNumStreams.restype = c_uint
        # C原型：int __stdcall MV_FG_GetNumStreams(IN DEVHANDLE hDevice, OUT unsigned int *pnNumStreams);
        return MVFGCtrldll.MV_FG_GetNumStreams(self.DEVHANDLE, byref(nNumStreams))

class FGStream():
    def __init__(self):
        self._STREAMHANDLE = c_void_p()  # 取流句柄
        self.STREAMHANDLE = pointer(self._STREAMHANDLE)  # 创建句柄指针
        
        self._BUFFERHANDLE = c_void_p()  # 缓存句柄
        self.BUFFERHANDLE = pointer(self._BUFFERHANDLE)  # 创建句柄指针

    def GetHandle(self):
        return self.STREAMHANDLE

    # ch:打开流通道 | en:Open Stream
    def OpenStream(self, hDevice, nIndex):
        handle = hDevice.GetHandle()
        MVFGCtrldll.MV_FG_OpenStream.argtype = (c_void_p, c_uint, c_void_p)
        MVFGCtrldll.MV_FG_OpenStream.restype = c_uint
        # C原型：int __stdcall MV_FG_OpenStream(IN DEVHANDLE hDevice, IN unsigned int nIndex, OUT STREAMHANDLE* phStream);
        return MVFGCtrldll.MV_FG_OpenStream(handle, nIndex, byref(self.STREAMHANDLE))

    # ch:关闭流通道 | en:Close Stream
    def CloseStream(self):
        MVFGCtrldll.MV_FG_CloseStream.argtype = c_void_p
        MVFGCtrldll.MV_FG_CloseStream.restype = c_uint
        # C原型：int __stdcall MV_FG_CloseStream(IN STREAMHANDLE hStream);
        return MVFGCtrldll.MV_FG_CloseStream(self.STREAMHANDLE)

    # ch:设置SDK内部缓存数量 | en:Set the Number of SDK internal Buffers
    def SetBufferNum(self, nBufferNum):
        MVFGCtrldll.MV_FG_SetBufferNum.argtype = (c_void_p, c_uint)
        MVFGCtrldll.MV_FG_SetBufferNum.restype = c_uint
        # C原型：int __stdcall MV_FG_SetBufferNum(IN STREAMHANDLE hStream, IN unsigned int nBufferNum);
        return MVFGCtrldll.MV_FG_SetBufferNum(self.STREAMHANDLE, nBufferNum)
        
    # ch:设置取流策略 | en:Set grab strategy
    def SetGrabStrategy(self, enGrabStrategy, bUseTrashBuffer):
        MVFGCtrldll.MV_FG_SetGrabStrategy.argtype = (c_void_p, c_int, c_bool)
        MVFGCtrldll.MV_FG_SetGrabStrategy.restype = c_uint
        # C原型：MV_FGCTRL_API int __stdcall MV_FG_SetGrabStrategy(IN STREAMHANDLE hStream, IN MV_FG_GRAB_STRATEGY enGrabStrategy, bool8_t bUseTrashBuffer);
        return MVFGCtrldll.MV_FG_SetGrabStrategy(self.DEVHANDLE, enGrabStrategy, bUseTrashBuffer)

    # ch:注册帧缓存信息回调函数(SDK内部申请缓存方式) | en:Register the callback function for frame buffer information. This API is valid only when buffers are requested internally by the SDK.
    def RegisterFrameCallBack(self, cbFrame, pUser):
        MVFGCtrldll.MV_FG_RegisterFrameCallBack.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_RegisterFrameCallBack.restype = c_uint
        # C原型：int __stdcall MV_FG_RegisterFrameCallBack(IN STREAMHANDLE hStream, IN MV_FG_FrameCallBack cbFrame, IN void* pUser);
        return MVFGCtrldll.MV_FG_RegisterFrameCallBack(self.STREAMHANDLE, cbFrame, pUser)
        
    # ch:注册帧缓存信息回调函数(SDK内部申请缓存方式) | en:Register the callback function for frame buffer information. This API is valid only when buffers are requested internally by the SDK.
    def RegisterFrameCallBackEx(self, cbFrame, pUser, bBufferRelease):
        MVFGCtrldll.MV_FG_RegisterFrameCallBackEx.argtype = (c_void_p, c_void_p, c_void_p, c_bool)
        MVFGCtrldll.MV_FG_RegisterFrameCallBackEx.restype = c_uint
        # C原型：int __stdcall MV_FG_RegisterFrameCallBackEx(IN STREAMHANDLE hStream, IN MV_FG_FrameCallBack cbFrame, IN void* pUser, IN bool8_t bBufferRelease);
        return MVFGCtrldll.MV_FG_RegisterFrameCallBackEx(self.STREAMHANDLE, cbFrame, pUser, bBufferRelease)

    # ch:获取一帧图像的缓存信息(SDK内部申请缓存方式) | en:Get the Buffer Information of One Frame of Image (SDK Internal Application Buffer Method)
    def GetFrameBuffer(self, stBufferInfo, nTimeout):
        MVFGCtrldll.MV_FG_GetFrameBuffer.argtype = (c_void_p, c_void_p, c_uint)
        MVFGCtrldll.MV_FG_GetFrameBuffer.restype = c_uint
        # C原型：int __stdcall MV_FG_GetFrameBuffer(IN STREAMHANDLE hStream, OUT MV_FG_BUFFER_INFO* pstBufferInfo, IN unsigned int nTimeout);
        return MVFGCtrldll.MV_FG_GetFrameBuffer(self.STREAMHANDLE, byref(stBufferInfo), nTimeout)

    # ch:释放缓存信息(SDK内部申请缓存方式，此接口用于释放不再使用的图像缓存，与MV_FG_GetFrameBuffer配套使用)
    # en:Release Buffer Information (SDK Internal Application Buffer Method, this interface can free image buffer, used with MV_FG_GetFrameBuffer)
    def ReleaseFrameBuffer(self, stBufferInfo):
        MVFGCtrldll.MV_FG_ReleaseFrameBuffer.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_ReleaseFrameBuffer.restype = c_uint
        # C原型：int __stdcall MV_FG_ReleaseFrameBuffer(IN STREAMHANDLE hStream, IN MV_FG_BUFFER_INFO* pstBufferInfo);
        return MVFGCtrldll.MV_FG_ReleaseFrameBuffer(self.STREAMHANDLE, byref(stBufferInfo))

    # ch:获取缓存内的ChunkData信息 | en:Get the ChunkData Information in the Buffer
    def GetBufferChunkData(self, stBufferInfo, nIndex, stChunkDataInfo):
        MVFGCtrldll.MV_FG_GetBufferChunkData.argtype = (c_void_p, c_void_p, c_uint, c_void_p)
        MVFGCtrldll.MV_FG_GetBufferChunkData.restype = c_uint
        # C原型：int __stdcall MV_FG_GetBufferChunkData(IN STREAMHANDLE hStream, IN MV_FG_BUFFER_INFO* pstBufferInfo, IN unsigned int nIndex, OUT MV_FG_CHUNK_DATA_INFO* pstChunkDataInfo);
        return MVFGCtrldll.MV_FG_GetBufferChunkData(self.STREAMHANDLE, byref(stBufferInfo), nIndex, byref(stChunkDataInfo))

    # ch:获取流通道的图像大小 | en:Get the Payload Size of the Stream
    def GetPayloadSize(self, nPayloadSize):
        MVFGCtrldll.MV_FG_GetPayloadSize.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_GetPayloadSize.restype = c_uint
        # C原型：int __stdcall MV_FG_GetPayloadSize(IN STREAMHANDLE hStream, OUT unsigned int* pnPayloadSize);
        return MVFGCtrldll.MV_FG_GetPayloadSize(self.STREAMHANDLE, byref(nPayloadSize))

    # ch:向流通道中注册缓存(必须在开始取流前注册缓存) | en:Register Buffer for Stream (the Buffer Must be Registered Before Starting Acquisition)
    def AnnounceBuffer(self, pBuffer, nSize, pPrivate):
        MVFGCtrldll.MV_FG_AnnounceBuffer.argtype = (c_void_p, c_void_p, c_uint, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_AnnounceBuffer.restype = c_uint
        # C原型：int __stdcall MV_FG_AnnounceBuffer(IN STREAMHANDLE hStream, IN void *pBuffer, IN unsigned int nSize, IN void *pPrivate, OUT BUFFERHANDLE *phBuffer);
        return MVFGCtrldll.MV_FG_AnnounceBuffer(self.STREAMHANDLE, pBuffer, nSize, pPrivate, byref(self.BUFFERHANDLE))
        
    # ch:从流通道中撤销缓存 | en:Revoke Buffer from Stream
    def RevokeBuffer(self, pBuffer, pPrivate):
        MVFGCtrldll.MV_FG_RevokeBuffer.argtype = (c_void_p, c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_RevokeBuffer.restype = c_uint
        # C原型：int __stdcall MV_FG_RevokeBuffer(IN STREAMHANDLE hStream, IN BUFFERHANDLE hBuffer, OUT void **pBuffer, OUT void **pPrivate);
        return MVFGCtrldll.MV_FG_RevokeBuffer(self.STREAMHANDLE, self.BUFFERHANDLE, byref(pBuffer), byref(pPrivate))
        
    # ch:刷新缓存队列 | en:Flush Buffer Queue
    def FlushQueue(self, enQueueType):
        MVFGCtrldll.MV_FG_FlushQueue.argtype = (c_void_p, c_int)
        MVFGCtrldll.MV_FG_FlushQueue.restype = c_uint
        # C原型：int __stdcall MV_FG_FlushQueue(IN STREAMHANDLE hStream, IN MV_FG_BUFFER_QUEUE_TYPE enQueueType);
        return MVFGCtrldll.MV_FG_FlushQueue(self.STREAMHANDLE, enQueueType)

    # ch:开始取流 | en:Start Acquisition
    def StartAcquisition(self):
        MVFGCtrldll.MV_FG_StartAcquisition.argtype = c_void_p
        MVFGCtrldll.MV_FG_StartAcquisition.restype = c_uint
        # C原型：int __stdcall MV_FG_StartAcquisition(IN STREAMHANDLE hStream);
        return MVFGCtrldll.MV_FG_StartAcquisition(self.STREAMHANDLE)

    # ch:停止取流 | en:Stop Acquisition
    def StopAcquisition(self):
        MVFGCtrldll.MV_FG_StopAcquisition.argtype = c_void_p
        MVFGCtrldll.MV_FG_StopAcquisition.restype = c_uint
        # C原型：int __stdcall MV_FG_StopAcquisition(IN STREAMHANDLE hStream);
        return MVFGCtrldll.MV_FG_StopAcquisition(self.STREAMHANDLE)

    # ch:获取一帧图像的缓存信息(用户向流通道注册缓存方式) | en:Get the Buffer Information of One Frame of Image (The User Registers the Buffer Method)
    def GetImageBuffer(self, nTimeout):
        MVFGCtrldll.MV_FG_GetImageBuffer.argtype = (c_void_p, c_void_p, c_uint)
        MVFGCtrldll.MV_FG_GetImageBuffer.restype = c_uint
        # C原型：int __stdcall MV_FG_GetImageBuffer(IN STREAMHANDLE hStream, OUT BUFFERHANDLE *phBuffer, IN unsigned int nTimeout);
        return MVFGCtrldll.MV_FG_GetImageBuffer(self.STREAMHANDLE, byref(self.BUFFERHANDLE), nTimeout)

    # ch:通过缓存句柄获取缓存信息 | en:Update Interface List
    def GetBufferInfo(self, stBufferInfo):
        MVFGCtrldll.MV_FG_GetBufferInfo.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_GetBufferInfo.restype = c_uint
        # C原型：int __stdcall MV_FG_GetBufferInfo(IN BUFFERHANDLE hBuffer, OUT MV_FG_BUFFER_INFO* pstBufferInfo);
        return MVFGCtrldll.MV_FG_GetBufferInfo(self.BUFFERHANDLE, byref(stBufferInfo))
        
    # ch:将缓存句柄放回输入队列 | en:Update Interface List
    def QueueBuffer(self):
        MVFGCtrldll.MV_FG_QueueBuffer.argtype = c_void_p
        MVFGCtrldll.MV_FG_QueueBuffer.restype = c_uint
        # C原型：int __stdcall MV_FG_QueueBuffer(IN BUFFERHANDLE hBuffer);
        return MVFGCtrldll.MV_FG_QueueBuffer(self.BUFFERHANDLE)

class FGImageProcess():
    def __init__(self, HANDLE):
        self._IMAGEHANDLE = c_void_p()  # 缓存句柄
        self.IMAGEHANDLE = pointer(self._IMAGEHANDLE)  # 创建句柄指针
        self.IMAGEHANDLE = HANDLE.GetHandle()

    # ch:显示一帧图像 | en:Display One Frame of Image
    def DisplayOneFrame(self, hWnd, stDisplayFrameInfo):
        MVFGCtrldll.MV_FG_DisplayOneFrame.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_DisplayOneFrame.restype = c_uint
        # C原型：int __stdcall MV_FG_DisplayOneFrame(IN IMAGEHANDLE hImage, IN void* hWnd, IN MV_FG_DISPLAY_FRAME_INFO *pstDisplayFrameInfo);
        return MVFGCtrldll.MV_FG_DisplayOneFrame(self.IMAGEHANDLE, hWnd, byref(stDisplayFrameInfo))
        
    # ch:在图像上绘制矩形框 | en:Draw Rect
    def DrawRect(self, stRectInfo):
        MVFGCtrldll.MV_FG_DrawRect.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_DrawRect.restype = c_uint
        # C原型：int __stdcall MV_FG_DrawRect(IN IMAGEHANDLE hImage, IN MVFG_RECT_INFO* pRectInfo);
        return MVFGCtrldll.MV_FG_DrawRect(self.IMAGEHANDLE, byref(stRectInfo))

    # ch:在图像上绘制圆形 | en:Draw Circle
    def DrawCircle(self, stCircleInfo):
        MVFGCtrldll.MV_FG_DrawCircle.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_DrawCircle.restype = c_uint
        # C原型：int __stdcall MV_FG_DrawCircle(IN IMAGEHANDLE hImage, IN MVFG_CIRCLE_INFO* pCircleInfo);
        return MVFGCtrldll.MV_FG_DrawCircle(self.IMAGEHANDLE, byref(stCircleInfo))

    # ch:在图像上绘制线条 | en:Draw Line
    def DrawLines(self, stLinesInfo):
        MVFGCtrldll.MV_FG_DrawLines.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_DrawLines.restype = c_uint
        # C原型：int __stdcall MV_FG_DrawLines(IN IMAGEHANDLE hImage, IN MVFG_LINES_INFO* pLinesInfo);
        return MVFGCtrldll.MV_FG_DrawLines(self.IMAGEHANDLE, byref(stLinesInfo))
        
    # ch:保存BMP图像 | en:Save BMP Image
    def SaveBitmap(self, stSaveBitmapInfo):
        MVFGCtrldll.MV_FG_SaveBitmap.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_SaveBitmap.restype = c_uint
        # C原型：int __stdcall MV_FG_SaveBitmap(IN IMAGEHANDLE hImage, IN OUT MV_FG_SAVE_BITMAP_INFO *pstSaveBitmapInfo):
        return MVFGCtrldll.MV_FG_SaveBitmap(self.IMAGEHANDLE, byref(stSaveBitmapInfo))

    # ch:保存JPEG图像 | en:Save JPEG Image
    def SaveJpeg(self, stSaveJpegInfo):
        MVFGCtrldll.MV_FG_SaveJpeg.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_SaveJpeg.restype = c_uint
        # C原型：int __stdcall MV_FG_SaveJpeg(IN IMAGEHANDLE hImage, IN OUT MV_FG_SAVE_JPEG_INFO *pstSaveJpegInfo);
        return MVFGCtrldll.MV_FG_SaveJpeg(self.IMAGEHANDLE, byref(stSaveJpegInfo))

    # ch:保存TIFF图像 | en:Save TIFF Image
    def SaveTiffToFile(self, stSaveTiffInfo):
        MVFGCtrldll.MV_FG_SaveTiffToFile.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_SaveTiffToFile.restype = c_uint
        # C原型：int __stdcall MV_FG_SaveTiffToFile(IN IMAGEHANDLE hImage, IN OUT MV_FG_SAVE_TIFF_TO_FILE_INFO *pstSaveTiffInfo);
        return MVFGCtrldll.MV_FG_SaveTiffToFile(self.IMAGEHANDLE, byref(stSaveTiffInfo))

    # ch:保存PNG图像 | Save PNG Image
    def SavePngToFile(self, stSavePngInfo):
        MVFGCtrldll.MV_FG_SavePngToFile.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_SavePngToFile.restype = c_uint
        # C原型：int __stdcall MV_FG_SavePngToFile(IN IMAGEHANDLE hImage, IN OUT MV_FG_SAVE_PNG_TO_FILE_INFO *pstSavePngInfo);
        return MVFGCtrldll.MV_FG_SavePngToFile(self.IMAGEHANDLE, byref(stSavePngInfo))

    # ch:像素格式转换 | en:Pixel Format Conversion
    def ConvertPixelType(self, stConvertPixelInfo):
        MVFGCtrldll.MV_FG_ConvertPixelType.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_ConvertPixelType.restype = c_uint
        # C原型：int __stdcall MV_FG_ConvertPixelType(IN IMAGEHANDLE hImage, IN OUT MV_FG_CONVERT_PIXEL_INFO *pstConvertPixelInfo);
        return MVFGCtrldll.MV_FG_ConvertPixelType(self.IMAGEHANDLE, byref(stConvertPixelInfo))

    # ch:无损解码 | en:High Bandwidth Decode
    def HB_Decode(self, stDecodeParam):
        MVFGCtrldll.MV_FG_HB_Decode.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_HB_Decode.restype = c_uint
        # C原型：int __stdcall MV_FG_HB_Decode(IN IMAGEHANDLE hImage, IN OUT MV_FG_HB_DECODE_PARAM* pstDecodeParam);
        return MVFGCtrldll.MV_FG_HB_Decode(self.IMAGEHANDLE, byref(stDecodeParam))

    # ch:JPEG解码 | en:JPEG Decoding
    def DecodeJpeg(self, stDecodeParam):
        MVFGCtrldll.MV_FG_DecodeJpeg.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_DecodeJpeg.restype = c_uint
        # C原型：int __stdcall MV_FG_DecodeJpeg(IN IMAGEHANDLE hImage, IN OUT MV_FG_DECODE_JPEG_PARAM* pstDecodeParam);
        return MVFGCtrldll.MV_FG_DecodeJpeg(self.IMAGEHANDLE, byref(stDecodeParam))

    # ch:重组图像 | en:Reconstruct Image
    def ReconstructImage(self, stReconstructInfo):
        MVFGCtrldll.MV_FG_ReconstructImageargtype.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_ReconstructImageargtype.restype = c_uint
        # C原型：int __stdcall MV_FG_ReconstructImage(IN IMAGEHANDLE hImage, IN OUT MV_FG_RECONSTRUCT_INFO *pstReconstructInfo);
        return MVFGCtrldll.MV_FG_ReconstructImageargtype(self.IMAGEHANDLE, byref(stReconstructInfo))

class FGGeneral():
    def __init__(self, HANDLE):
        self._PORTHANDLE = c_void_p()  # 通用句柄
        self.PORTHANDLE = pointer(self._PORTHANDLE)  # 创建句柄指针
        self.PORTHANDLE = HANDLE.GetHandle()

    # ch:获取XML文件 | en:Get XML File
    def GetXMLFile(self, strData, nDataSize, nDataLen):
        MVFGCtrldll.MV_FG_GetXMLFile.argtype = (c_void_p, c_void_p, c_uint, c_void_p)
        MVFGCtrldll.MV_FG_GetXMLFile.restype = c_uint
        # C原型：int __stdcall MV_FG_GetXMLFile(IN PORTHANDLE hPort, IN OUT unsigned char* pData, IN unsigned int nDataSize, OUT unsigned int* pnDataLen);
        return MVFGCtrldll.MV_FG_GetXMLFile(self.PORTHANDLE, strData, nDataSize, byref(nDataLen))

    # ch:获得节点的访问模式 | en:Get the Access Mode of the Node
    def GetNodeAccessMode(self, strName, enAccessMode):
        MVFGCtrldll.MV_FG_GetNodeAccessMode.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_GetNodeAccessMode.restype = c_uint
        # C原型：int __stdcall MV_FG_GetNodeAccessMode(IN PORTHANDLE hPort, IN const char * strName, OUT MV_FG_NODE_ACCESS_MODE *penAccessMode);
        return MVFGCtrldll.MV_FG_GetNodeAccessMode(self.PORTHANDLE, strName.encode('ascii'), byref(enAccessMode))

    # ch:获得节点的类型 | en:Get the Type of Node
    def GetNodeInterfaceType(self, strName, enInterfaceType):
        MVFGCtrldll.MV_FG_GetNodeInterfaceType.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_GetNodeInterfaceType.restype = c_uint
        # C原型：int __stdcall MV_FG_GetNodeInterfaceType(IN PORTHANDLE hPort, IN const char * strName, OUT MV_FG_NODE_INTERFACE_TYPE *penInterfaceType);
        return MVFGCtrldll.MV_FG_GetNodeInterfaceType(self.PORTHANDLE, strName.encode('ascii'), byref(enInterfaceType))

    # ch:获取整型节点信息 | en:Get Integer Node Information
    def GetIntValue(self, strKey, stIntValue):
        MVFGCtrldll.MV_FG_GetIntValue.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_GetIntValue.restype = c_uint
        # C原型：int __stdcall MV_FG_GetIntValue(IN PORTHANDLE hPort, IN const char* strKey, OUT MV_FG_INTVALUE *pstIntValue);
        return MVFGCtrldll.MV_FG_GetIntValue(self.PORTHANDLE, strKey.encode('ascii'), byref(stIntValue))

    # ch:设置整型节点信息 | en:Set Integer Node Information
    def SetIntValue(self, strKey, nValue):
        MVFGCtrldll.MV_FG_SetIntValue.argtype = (c_void_p, c_void_p, c_uint64)
        MVFGCtrldll.MV_FG_SetIntValue.restype = c_uint
        # C原型：int __stdcall MV_FG_SetIntValue(IN PORTHANDLE hPort, IN const char* strKey, IN int64_t nValue);
        return MVFGCtrldll.MV_FG_SetIntValue(self.PORTHANDLE, strKey.encode('ascii'), nValue)

    # ch:获取枚举类型节点的信息 | en:Get the Information of Enumeration Type Node
    def GetEnumValue(self, strKey, stEnumValue):
        MVFGCtrldll.MV_FG_GetEnumValue.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_GetEnumValue.restype = c_uint
        # C原型：int __stdcall MV_FG_GetEnumValue(IN PORTHANDLE hPort, IN const char* strKey, OUT MV_FG_ENUMVALUE *pstEnumValue);
        return MVFGCtrldll.MV_FG_GetEnumValue(self.PORTHANDLE, strKey.encode('ascii'), byref(stEnumValue))

    # ch:设置枚举类型节点的信息 | en:Set the Information of Enumeration Type Node
    def SetEnumValue(self, strKey, nValue):
        MVFGCtrldll.MV_FG_SetEnumValue.argtype = (c_void_p, c_void_p, c_uint)
        MVFGCtrldll.MV_FG_SetEnumValue.restype = c_uint
        # C原型：int __stdcall MV_FG_SetEnumValue(IN PORTHANDLE hPort, IN const char* strKey, IN unsigned int nValue);
        return MVFGCtrldll.MV_FG_SetEnumValue(self.PORTHANDLE, strKey.encode('ascii'), nValue)

    # ch:通过字符串设置枚举类型节点的信息 | en:Set the Information of Enumeration Type Node through string
    def SetEnumValueByString(self, strKey, strValue):
        MVFGCtrldll.MV_FG_SetEnumValueByString.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_SetEnumValueByString.restype = c_uint
        # C原型：int __stdcall MV_FG_SetEnumValueByString(IN PORTHANDLE hPort, IN const char* strKey, IN const char* strValue);
        return MVFGCtrldll.MV_FG_SetEnumValueByString(self.PORTHANDLE, strKey.encode('ascii'), strValue.encode('ascii'))

    # ch:获取单精度浮点型节点的信息 | en:Get the Information of the float Node
    def GetFloatValue(self, strKey, stFloatValue):
        MVFGCtrldll.MV_FG_GetFloatValue.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_GetFloatValue.restype = c_uint
        # C原型：int __stdcall MV_FG_GetFloatValue(IN PORTHANDLE hPort, IN const char* strKey, OUT MV_FG_FLOATVALUE *pstFloatValue);
        return MVFGCtrldll.MV_FG_GetFloatValue(self.PORTHANDLE, strKey.encode('ascii'), byref(stFloatValue))

    # ch:设置单精度浮点型节点的信息 | en:Set the Information of the float Node
    def SetFloatValue(self, strKey, fValue):
        MVFGCtrldll.MV_FG_GetFloatValue.argtype = (c_void_p, c_void_p, c_float)
        MVFGCtrldll.MV_FG_GetFloatValue.restype = c_uint
        # C原型：int __stdcall MV_FG_SetFloatValue(IN PORTHANDLE hPort, IN const char* strKey, IN float fValue);
        return MVFGCtrldll.MV_FG_GetFloatValue(self.PORTHANDLE, strKey.encode('ascii'), fValue)

    # ch:获取布尔型节点的信息 | en:Get Information about Boolean Node
    def GetBoolValue(self, strKey, bValue):
        MVFGCtrldll.MV_FG_GetBoolValue.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_GetBoolValue.restype = c_uint
        # C原型：int __stdcall MV_FG_GetBoolValue(IN PORTHANDLE hPort, IN const char* strKey, OUT bool8_t *pbValue);
        return MVFGCtrldll.MV_FG_GetBoolValue(self.PORTHANDLE, strKey.encode('ascii'), byref(bValue))

    # ch:设置布尔类型节点的信息 | en:Set Information about Boolean Node
    def SetBoolValue(self, strKey, bValue):
        MVFGCtrldll.V_FG_SetBoolValue.argtype = (c_void_p, c_void_p, c_bool)
        MVFGCtrldll.V_FG_SetBoolValue.restype = c_uint
        # C原型：int __stdcall MV_FG_SetBoolValue(IN PORTHANDLE hPort, IN const char* strKey, IN bool8_t bValue);
        return MVFGCtrldll.V_FG_SetBoolValue(self.PORTHANDLE, strKey.encode('ascii'), bValue)

    # ch:获取字符串型节点的信息 | en:Get the Information of string Node
    def GetStringValue(self, strKey, stStringValue):
        MVFGCtrldll.MV_FG_GetStringValueargtype.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_GetStringValueargtype.restype = c_uint
        # C原型：int __stdcall MV_FG_GetStringValue(IN PORTHANDLE hPort, IN const char* strKey, OUT MV_FG_STRINGVALUE *pstStringValue);
        return MVFGCtrldll.MV_FG_GetStringValueargtype(self.PORTHANDLE, strKey.encode('ascii'), byref(stStringValue))

    # ch:设置字符串型节点的信息 | en:Set the Information of string Node
    def SetStringValue(self, strKey, strValue):
        MVFGCtrldll.MV_FG_SetStringValue.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_SetStringValue.restype = c_uint
        # C原型：int __stdcall MV_FG_SetStringValue(IN PORTHANDLE hPort, IN const char* strKey, IN const char* strValue);
        return MVFGCtrldll.MV_FG_SetStringValue(self.PORTHANDLE, strKey.encode('ascii'), strValue.encode('ascii'))

    # ch:执行命令型节点的命令 | en:Execute Command of Command Node
    def SetCommandValue(self, strKey):
        MVFGCtrldll.MV_FG_SetCommandValue.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_SetCommandValue.restype = c_uint
        # C原型：int __stdcall MV_FG_SetCommandValue(IN PORTHANDLE hPort, IN const char* strKey);
        return MVFGCtrldll.MV_FG_SetCommandValue(self.PORTHANDLE, strKey.encode('ascii'))

    # ch:配置自定义的整型值 | en:Configure Custom Integer Values
    def SetConfigIntValue(self, enConfigCmd, nValue):
        MVFGCtrldll.MV_FG_SetConfigIntValue.argtype = (c_void_p, c_int, c_uint64)
        MVFGCtrldll.MV_FG_SetConfigIntValue.restype = c_uint
        # C原型：int __stdcall MV_FG_SetConfigIntValue(IN PORTHANDLE hPort, IN MV_FG_CONFIG_CMD enConfigCmd, IN int64_t nValue);
        return MVFGCtrldll.MV_FG_SetConfigIntValue(self.PORTHANDLE, enConfigCmd, nValue)

    # ch:保存设备属性 | en:Save device feature
    def FeatureSave(self, strFileName):
        MVFGCtrldll.MV_FG_FeatureSave.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_FeatureSave.restype = c_uint
        # C原型：int __stdcall MV_FG_FeatureSave(IN PORTHANDLE hPort, IN const char* strFileName);
        return MVFGCtrldll.MV_FG_FeatureSave(self.PORTHANDLE, strFileName.encode('ascii'))

    # ch:导入设备属性 | en:Load device feature
    def FeatureLoad(self, strFileName):
        MVFGCtrldll.MV_FG_FeatureLoad.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_FeatureLoad.restype = c_uint
        # C原型：int __stdcall MV_FG_FeatureLoad(IN PORTHANDLE hPort, IN const char* strFileName);
        return MVFGCtrldll.MV_FG_FeatureLoad(self.PORTHANDLE, strFileName.encode('ascii'))

    # ch:读取设备寄存器 | en:Read Memory
    def ReadPort(self, pBuffer, nAddress, nLength):
        MVFGCtrldll.MV_FG_ReadPort.argtype = (c_void_p, c_void_p, c_int64, c_int64)
        MVFGCtrldll.MV_FG_ReadPort.restype = c_uint
        # C原型：int __stdcall MV_FG_ReadPort(IN PORTHANDLE hPort, void *pBuffer, int64_t nAddress, int64_t nLength);
        return MVFGCtrldll.MV_FG_ReadPort(self.PORTHANDLE, pBuffer, nAddress, nLength)

    # ch:写入设备寄存器 | en:Write Memory
    def WritePort(self, pBuffer, nAddress, nLength):
        MVFGCtrldll.MV_FG_WritePort.argtype = (c_void_p, c_void_p, c_int64, c_int64)
        MVFGCtrldll.MV_FG_WritePort.restype = c_uint
        # C原型：int __stdcall MV_FG_ReadPort(IN PORTHANDLE hPort, void *pBuffer, int64_t nAddress, int64_t nLength);
        return MVFGCtrldll.MV_FG_WritePort(self.PORTHANDLE, pBuffer, nAddress, nLength)

    # ch:从设备读取文件 | en:Read the file data from device
    def FileAccessRead(self, stFileAccess):
        MVFGCtrldll.MV_FG_FileAccessRead.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_FileAccessRead.restype = c_uint
        # C原型：int __stdcall MV_FG_FileAccessRead(IN PORTHANDLE hPort, IN OUT MV_FG_FILE_ACCESS* pstFileAccess);
        return MVFGCtrldll.MV_FG_FileAccessRead(self.PORTHANDLE, byref(stFileAccess))

    # ch:将文件写入设备 | en:Write the file to device
    def FileAccessWrite(self, stFileAccess):
        MVFGCtrldll.MV_FG_FileAccessWrite.argtype = (c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_FileAccessWrite.restype = c_uint
        # C原型：int __stdcall MV_FG_FileAccessWrite(IN PORTHANDLE hPort, IN MV_FG_FILE_ACCESS* pstFileAccess);
        return MVFGCtrldll.MV_FG_FileAccessWrite(self.PORTHANDLE, byref(stFileAccess))

    # ch:清除GenICam节点缓存 | en:Invalidate GenICam Nodes
    def InvalidateNodes(self):
        MVFGCtrldll.MV_FG_InvalidateNodes.argtype = c_void_p
        MVFGCtrldll.MV_FG_InvalidateNodes.restype = c_uint
        # C原型：int __stdcall MV_FG_InvalidateNodes(IN PORTHANDLE hPort);
        return MVFGCtrldll.MV_FG_InvalidateNodes(self.PORTHANDLE)
        
class FGEvent():
    def __init__(self, HANDLE):
        self._EVENTHANDLE = c_void_p()  # 事件句柄
        self.EVENTHANDLE = pointer(self._EVENTHANDLE)  # 创建句柄指针
        self.EVENTHANDLE = HANDLE.GetHandle()

    # ch:注册事件回调函数 | en:Register Event Callback Function
    def RegisterEventCallBack(self, strEventName, cbEvent, pUser):
        MVFGCtrldll.MV_FG_RegisterEventCallBack.argtype = (c_void_p, c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_RegisterEventCallBack.restype = c_uint
        # C原型：int __stdcall MV_FG_RegisterEventCallBack(IN PORTHANDLE hPort, IN const char* strEventName, IN MV_FG_EventCallBack cbEvent, IN void* pUser);
        return MVFGCtrldll.MV_FG_RegisterEventCallBack(self.EVENTHANDLE, strEventName.encode('ascii'), cbEvent, pUser)

    # ch:注册异常信息回调函数 | en:Register Exception Information Callback Function
    def RegisterExceptionCallBack(self, cbException, pUser):
        MVFGCtrldll.MV_FG_RegisterExceptionCallBack.argtype = (c_void_p, c_void_p, c_void_p)
        MVFGCtrldll.MV_FG_RegisterExceptionCallBack.restype = c_uint
        # C原型：int __stdcall MV_FG_RegisterExceptionCallBack(IN PORTHANDLE hPort, IN MV_FG_ExceptionCallBack cbException, IN void* pUser);
        return MVFGCtrldll.MV_FG_RegisterExceptionCallBack(self.EVENTHANDLE, cbException, pUser)