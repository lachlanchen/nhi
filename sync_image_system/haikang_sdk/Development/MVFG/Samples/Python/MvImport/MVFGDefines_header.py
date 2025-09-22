
from ctypes import *
from enum import Enum
from MVFGDefines_const import *

MV_FG_PIXEL_TYPE       = c_int # enum
MV_FG_CFA_METHOD       = c_int # enum
MV_FG_RESOLUTION_UNIT  = c_int # enum
MV_FG_GAMMA_TYPE       = c_int # enum
MV_FG_RECONSTRUCT_MODE = c_int # enum


# < \~chinese CXP采集卡信息         \~english CXP interface information
class _MV_CXP_INTERFACE_INFO_(Structure):
    pass
_MV_CXP_INTERFACE_INFO_._fields_ = [
    ('chInterfaceID', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),     # < \~chinese 采集卡ID                 \~english Interface ID
    ('chDisplayName', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),     # < \~chinese 显示名称                 \~english Display name
    ('chSerialNumber', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),    # < \~chinese 序列号                   \~english Serial number
    ('nPCIEInfo', c_uint),                                   # < \~chinese 采集卡的PCIE插槽信息       \~english PCIe slot information of interface
    ('chModelName', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),       # <  \~chinese Interface型号           \~english Interface model name
    ('chManufacturer', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),    # < \~chinese Interface厂商            \~english Interface manufacturer name
    ('chDeviceVersion', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),   # < \~chinese Interface版本            \~english Interface device version
    ('chUserDefinedName', c_ubyte * MV_FG_MAX_IF_INFO_SIZE), # < \~chinese Interface自定义名称       \~english Interface user defined name

]
MV_CXP_INTERFACE_INFO = _MV_CXP_INTERFACE_INFO_

# < \~chinese GEV采集卡信息         \~english GEV interface information
class _MV_GEV_INTERFACE_INFO_(Structure):
    pass
_MV_GEV_INTERFACE_INFO_._fields_ = [
    ('chInterfaceID', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),     # < \~chinese 采集卡ID                \~english Interface ID
    ('chDisplayName', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),     # < \~chinese 显示名称                 \~english Display name
    ('chSerialNumber', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),    # < \~chinese 序列号                   \~english Serial number
    ('nPCIEInfo', c_uint),                                   # < \~chinese 采集卡的PCIE插槽信息       \~english PCIe slot information of interface
    ('chModelName', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),       # <  \~chinese Interface型号           \~english Interface model name
    ('chManufacturer', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),    # < \~chinese Interface厂商            \~english Interface manufacturer name
    ('chDeviceVersion', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),   # < \~chinese Interface版本            \~english Interface device version
    ('chUserDefinedName', c_ubyte * MV_FG_MAX_IF_INFO_SIZE), # < \~chinese Interface自定义名称       \~english Interface user defined name
]
MV_GEV_INTERFACE_INFO = _MV_GEV_INTERFACE_INFO_

# < \~chinese CML采集卡信息         \~english CameraLink interface information
class _MV_CML_INTERFACE_INFO_(Structure):
    pass
_MV_CML_INTERFACE_INFO_._fields_ = [
    ('chInterfaceID', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),     # < \~chinese 采集卡ID                \~english Interface ID
    ('chDisplayName', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),     # < \~chinese 显示名称                 \~english Display name
    ('chSerialNumber', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),    # < \~chinese 序列号                   \~english Serial number
    ('nPCIEInfo', c_uint),                                   # < \~chinese 采集卡的PCIE插槽信息       \~english PCIe slot information of interface
    ('chModelName', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),       # <  \~chinese Interface型号           \~english Interface model name
    ('chManufacturer', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),    # < \~chinese Interface厂商            \~english Interface manufacturer name
    ('chDeviceVersion', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),   # < \~chinese Interface版本            \~english Interface device version
    ('chUserDefinedName', c_ubyte * MV_FG_MAX_IF_INFO_SIZE), # < \~chinese Interface自定义名称       \~english Interface user defined name
]
MV_CML_INTERFACE_INFO = _MV_CML_INTERFACE_INFO_

# < \~chinese XoF采集卡信息         \~english XoF interface information
class _MV_XoF_INTERFACE_INFO_(Structure):
    pass
_MV_XoF_INTERFACE_INFO_._fields_ = [
    ('chInterfaceID', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),     # < \~chinese 采集卡ID                 \~english Interface ID
    ('chDisplayName', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),     # < \~chinese 显示名称                 \~english Display name
    ('chSerialNumber', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),    # < \~chinese 序列号                   \~english Serial number
    ('nPCIEInfo', c_uint),                                   # < \~chinese 采集卡的PCIE插槽信息       \~english PCIe slot information of interface
    ('chModelName', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),       # <  \~chinese Interface型号           \~english Interface model name
    ('chManufacturer', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),    # < \~chinese Interface厂商            \~english Interface manufacturer name
    ('chDeviceVersion', c_ubyte * MV_FG_MAX_IF_INFO_SIZE),   # < \~chinese Interface版本            \~english Interface device version
    ('chUserDefinedName', c_ubyte * MV_FG_MAX_IF_INFO_SIZE), # < \~chinese Interface自定义名称       \~english Interface user defined name

]
MV_XoF_INTERFACE_INFO = _MV_XoF_INTERFACE_INFO_

# < \~chinese 采集卡信息            \~english Interface information
class _MV_FG_INTERFACE_INFO_(Structure):
    pass
class MV_FG_INTERFACE_INFO_UNION(Union):
    pass
MV_FG_INTERFACE_INFO_UNION._fields_ = [
    ('stCXPIfaceInfo', MV_CXP_INTERFACE_INFO),             # < \~chinese CXP采集卡信息         \~english CXP interface information
    ('stGEVIfaceInfo', MV_GEV_INTERFACE_INFO),             # < \~chinese GEV采集卡信息         \~english GEV interface information
    ('stCMLIfaceInfo', MV_CML_INTERFACE_INFO),             # < \~chinese CameraLink采集卡信息   \~english CML interface information
    ('stXoFIfaceInfo', MV_XoF_INTERFACE_INFO),             # < \~chinese XoF采集卡信息          \~english XoF interface information
    ('nReserved', c_uint * 256),                           # < \~chinese 保留字段               \~english Reserved
]

_MV_FG_INTERFACE_INFO_._fields_ = [
    ('nTLayerType', c_uint),                                # < \~chinese 采集卡类型    \~english Interface type
    ('nReserved', c_uint * 4),                              # < \~chinese 保留字段      \~english Reserved
    ('IfaceInfo', MV_FG_INTERFACE_INFO_UNION),

]
MV_FG_INTERFACE_INFO = _MV_FG_INTERFACE_INFO_

# < \~chinese CXP设备信息      \~english CXP device information
class _MV_CXP_DEVICE_INFO_(Structure):
    pass
_MV_CXP_DEVICE_INFO_._fields_ = [
    ('chVendorName', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),         # < \~chinese 供应商名字       \~english Vendor name
    ('chModelName', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),          # < \~chinese 型号名字         \~english Model name
    ('chManufacturerInfo', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),   # < \~chinese 厂商信息         \~english Manufacturer information
    ('chDeviceVersion', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),      # < \~chinese 设备版本         \~english Device version
    ('chSerialNumber', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),       # < \~chinese 序列号           \~english Serial number
    ('chUserDefinedName', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),    # < \~chinese 用户自定义名字   \~english User defined name
    ('chDeviceID', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),           # < \~chinese 设备ID            \~english Device ID
    ('nReserved', c_uint * 48),                                 # < \~chinese 保留字段      \~english Reserved
]
MV_CXP_DEVICE_INFO = _MV_CXP_DEVICE_INFO_

# < \~chinese GEV设备信息          \~english GEV device information
class _MV_GEV_DEVICE_INFO_(Structure):
    pass
_MV_GEV_DEVICE_INFO_._fields_ = [
    ('nIpCfgOption', c_uint),                         # < \~chinese 支持的IP配置       \~english Supported IP configurations
    ('nIpCfgCurrent', c_uint),                        # < \~chinese 当前IP配置，参考支持的IP配置说明   \~english For the current IP configuration, refer to the supported IP configuration description

    ('nCurrentIp', c_uint),                           # < \~chinese 当前IP地址       \~english Current IP address
    ('nCurrentSubNetMask', c_uint),                   # < \~chinese 当前子网掩码     \~english Current subnet mask
    ('nDefultGateWay', c_uint),                       # < \~chinese 当前网关         \~english Current gateway
    ('nNetExport', c_uint),                           # < \~chinese 网口IP地址       \~english Network port IP address
    ('nMacAddress', c_uint64),                        # < \~chinese MAC地址          \~english MAC address

    ('chVendorName', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),            # < \~chinese 供应商名字       \~english Vendor name
    ('chModelName', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),             # < \~chinese 型号名字         \~english Model name
    ('chManufacturerInfo', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),      # < \~chinese 厂商信息         \~english Manufacturer information
    ('chDeviceVersion', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),         # < \~chinese 设备版本         \~english Device version
    ('chSerialNumber', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),          # < \~chinese 序列号           \~english Serial number
    ('chUserDefinedName', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),       # < \~chinese 用户自定义名字   \~english User defined name
    ('chDeviceID', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),              # < \~chinese 设备ID           \~english Device ID

    ('nReserved', c_uint * 48),                                    # < \~chinese 保留字段         \~english Reserved

]
MV_GEV_DEVICE_INFO = _MV_GEV_DEVICE_INFO_

# < \~chinese CML设备信息          \~english CML device information
class _MV_CML_DEVICE_INFO_(Structure):
    pass
_MV_CML_DEVICE_INFO_._fields_ = [
    ('chVendorName', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),         # < \~chinese 供应商名字       \~english Vendor name
    ('chModelName', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),          # < \~chinese 型号名字         \~english Model name
    ('chManufacturerInfo', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),   # < \~chinese 厂商信息         \~english Manufacturer information
    ('chDeviceVersion', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),      # < \~chinese 设备版本         \~english Device version
    ('chSerialNumber', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),       # < \~chinese 序列号           \~english Serial number
    ('chUserDefinedName', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),    # < \~chinese 用户自定义名字   \~english User defined name
    ('chDeviceID', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),           # < \~chinese 设备ID            \~english Device ID

    ('nReserved', c_uint * 48),                                 # < \~chinese 保留字段         \~english Reserved
]
MV_CML_DEVICE_INFO = _MV_CML_DEVICE_INFO_

# < \~chinese XoF设备信息         \~english XoF device information
class _MV_XoF_DEVICE_INFO_(Structure):
    pass
_MV_XoF_DEVICE_INFO_._fields_ = [
    ('chVendorName', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),         # < \~chinese 供应商名字       \~english Vendor name
    ('chModelName', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),          # < \~chinese 型号名字         \~english Model name
    ('chManufacturerInfo', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),   # < \~chinese 厂商信息         \~english Manufacturer information
    ('chDeviceVersion', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),      # < \~chinese 设备版本         \~english Device version
    ('chSerialNumber', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),       # < \~chinese 序列号           \~english Serial number
    ('chUserDefinedName', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),    # < \~chinese 用户自定义名字   \~english User defined name
    ('chDeviceID', c_ubyte * MV_FG_MAX_DEV_INFO_SIZE),           # < \~chinese 设备ID            \~english Device ID

    ('nReserved', c_uint * 48),                                 # < \~chinese 保留字段         \~english Reserved
]
MV_XoF_DEVICE_INFO = _MV_XoF_DEVICE_INFO_

# < \~chinese 设备信息         \~english Device information
class _MV_FG_DEVICE_INFO_(Structure):
    pass
class MV_FG_DEVICE_INFO_UNION(Union):
    pass
MV_FG_DEVICE_INFO_UNION._fields_ = [
    ('stCXPDevInfo', MV_CXP_DEVICE_INFO),                   # < \~chinese CXP设备信息          \~english CXP device information
    ('stGEVDevInfo', MV_GEV_DEVICE_INFO),                   # < \~chinese GEV设备信息          \~english GEV device information
    ('stCMLDevInfo', MV_CML_DEVICE_INFO),                  # < \~chinese CameraLink设备信息    \~english CameraLink device information
    ('stXoFDevInfo', MV_XoF_DEVICE_INFO),                  # < \~chinese XoF设备信息           \~english XoF device information

    ('nReserved', c_uint * 256),                            # < \~chinese 保留字段         \~english Reserved
]
_MV_FG_DEVICE_INFO_._fields_ = [
    ('nDevType', c_uint),                                   # < \~chinese 设备类型     \~english Device type
    ('nReserved', c_uint * 3),                              # < \~chinese 保留字段     \~english Reserved
    ('DevInfo', MV_FG_DEVICE_INFO_UNION),
]
MV_FG_DEVICE_INFO = _MV_FG_DEVICE_INFO_

# < \~chinese 输出的帧缓存信息          \~english Output Frame buffer information
class _MV_FG_BUFFER_INFO_(Structure):
    pass
_MV_FG_BUFFER_INFO_._fields_ = [
    ('pBuffer', c_void_p),                                  # < [OUT] \~chinese 图像缓存地址       \~english Image buffer address
    ('nSize', c_uint),                                      # < [OUT] \~chinese 地址大小           \~english Address size
    ('nFilledSize', c_uint),                                # < [OUT] \~chinese 图像长度           \~english Image length
    ('pPrivate', c_void_p),                                 # < [OUT] \~chinese 私有数据           \~english Private data

    ('nWidth', c_uint),                                     # < [OUT] \~chinese 宽度               \~english Width
    ('nHeight', c_uint),                                    # < [OUT] \~chinese 高度               \~english Height
    ('enPixelType', MV_FG_PIXEL_TYPE),                      # < [OUT] \~chinese 像素格式           \~english Pixel type

    ('bNewData', c_bool),                                   # < [OUT] \~chinese 是否是新图像到来   \~english Is it a new image coming
    ('bQueued', c_bool),                                    # < [OUT] \~chinese 是否在取图队列中   \~english Is it in the drawing queue
    ('bAcquiring', c_bool),                                 # < [OUT] \~chinese 是否在取图         \~english Is it taking pictures
    ('bIncomplete', c_bool),                                # < [OUT] \~chinese 是否未完成         \~english Is it incomplete

    ('nFrameID', c_int64),                                  # < [OUT] \~chinese 帧号               \~english Frame ID
    ('nDevTimeStamp', c_int64),                             # < [OUT] \~chinese 设备时间戳         \~english Device timestamp
    ('nHostTimeStamp', c_int64),                            # < [OUT] \~chinese 主机时间戳         \~english Host timestamp

    ('nNumChunks', c_uint),                                 # < [OUT] \~chinese Chunk个数          \~english Number of Chunks
    ('nChunkPayloadSize', c_uint),                          # < [OUT] \~chinese Chunk负载长度      \~english Payload size of Chunk

    ('nSecondCount', c_uint),                               #< [OUT] \~chinese 秒数(时标)         \~english Seconds (time scale)
    ('nCycleCount', c_uint),                                # < [OUT] \~chinese 周期数(时标)       \~english Number of cycles (time scale)
    ('nCycleOffset', c_uint),                               # < [OUT] \~chinese 周期偏移量(时标)   \~english Cycle offset (time scale)

    ('fGain', c_float),                                     # < [OUT] \~chinese 增益               \~english Gain
    ('fExposureTime', c_float),                             # < [OUT] \~chinese 曝光时间           \~english Exposure time
    ('nAverageBrightness', c_uint),                         # < [OUT] \~chinese 平均亮度           \~english Average brightness
    ('nFrameCounter', c_uint),                              # < [OUT] \~chinese 总帧数             \~english Total frames
    ('nTriggerIndex', c_uint),                              # < [OUT] \~chinese 触发计数           \~english Trigger index

    ('nInput', c_uint),                                     # < [OUT] \~chinese 输入               \~english Input
    ('nOutput', c_uint),                                    # < [OUT] \~chinese 输出               \~english Output

    ('nRed', c_uint),                                       # < [OUT] \~chinese 红色(白平衡)       \~english Red (white balance)
    ('nGreen', c_uint),                                     # < [OUT] \~chinese 绿色(白平衡)       \~english Green (white balance)
    ('nBlue', c_uint),                                      # < [OUT] \~chinese 蓝色(白平衡)       \~english Blue (white balance

    ('nOffsetX', c_uint),                                   # < [OUT] \~chinese 水平偏移量(ROI位置)   \~english Horizontal offset (ROI position)
    ('nOffsetY', c_uint),                                   # < [OUT] \~chinese 垂直偏移量(ROI位置)   \~english Vertical offset (ROI position)
    ('nChunkWidth', c_uint),                                # < [OUT] \~chinese 宽度(ROI位置)         \~english Width (ROI position)
    ('nChunkHeight', c_uint),                               # < [OUT] \~chinese 高度(ROI位置)         \~english Height (ROI position)

    ('nLastFrameFlag', c_uint),                             # < [OUT] \~chinese 电平结束最后一帧   \~english last level frame flag
    
    ('nReserved', c_uint * 44),                             # < \~chinese 保留字段     \~english Reserved
]
MV_FG_BUFFER_INFO = _MV_FG_BUFFER_INFO_

# < \~chinese Chunk信息          \~english Chunk information
class _MV_FG_CHUNK_DATA_INFO_(Structure):
    pass
_MV_FG_CHUNK_DATA_INFO_._fields_ = [
    ('pChunkData', POINTER(c_ubyte)),                                       # < [OUT] \~chinese Chunk数据    \~english Chunk data
    ('nChunkID', c_uint),                                                   # < [OUT] \~chinese Chunk ID     \~english Chunk ID
    ('nChunkLen', c_uint),                                                  # < [OUT] \~chinese Chunk的长度  \~english Length of Chunk

    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved
]
MV_FG_CHUNK_DATA_INFO = _MV_FG_CHUNK_DATA_INFO_

# \~chinese 输入的图像信息           \~english Input image information
class _MV_FG_INPUT_IMAGE_INFO_(Structure):
    pass
_MV_FG_INPUT_IMAGE_INFO_._fields_ = [
    ('nWidth', c_uint),                                                      # < [IN]   \~chinese 图像宽       \~english Width
    ('nHeight', c_uint),                                                     # < [IN]   \~chinese 图像高       \~english Height
    ('enPixelType',  MV_FG_PIXEL_TYPE),                                      # < [IN]   \~chinese 像素格式     \~english Pixel type

    ('pImageBuf', POINTER(c_ubyte)),                                         # < [IN]   \~chinese 输入图像缓存    \~english Input image buffer
    ('nImageBufLen', c_uint),                                                # < [IN]   \~chinese 输入图像长度    \~english Input image length

    ('nReserved', c_uint * 4),                                               # < \~chinese 保留字段     \~english Reserved

]
MV_FG_INPUT_IMAGE_INFO   = _MV_FG_INPUT_IMAGE_INFO_
MV_FG_DISPLAY_FRAME_INFO = _MV_FG_INPUT_IMAGE_INFO_

# \~chinese 输出的图像信息           \~english Output image information
class _MV_FG_OUTPUT_IMAGE_INFO_(Structure):
    pass
_MV_FG_OUTPUT_IMAGE_INFO_._fields_ = [
    ('nWidth', c_uint),                                                     # < [OUT]   \~chinese 图像宽       \~english Width
    ('nHeight', c_uint),                                                    # < [OUT]   \~chinese 图像高       \~english Height
    ('enPixelType',  MV_FG_PIXEL_TYPE),                                     # < [OUT]   \~chinese 像素格式     \~english Pixel type

    ('pImageBuf', POINTER(c_ubyte)),                                        # < [IN][OUT]   \~chinese 输入图像缓存    \~english Input image buffer
    ('nImageBufLen', c_uint),                                               # < [IN]   \~chinese 输入图像长度    \~english Input image length
    ('nImageBufLen', c_uint),                                               # < [OUT]  \~chinese 图像长度      \~english Image length

    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved

]
MV_FG_OUTPUT_IMAGE_INFO = _MV_FG_OUTPUT_IMAGE_INFO_

# < \~chinese 保存BMP图像信息         \~english Save BMP image information
class _MV_FG_SAVE_BITMAP_INFO_(Structure):
    pass
_MV_FG_SAVE_BITMAP_INFO_._fields_ = [
    ('stInputImageInfo', MV_FG_INPUT_IMAGE_INFO),                           # < [IN]  \~chinese 输入图片信息     \~english Input image information
    ('pBmpBuf', POINTER(c_ubyte)),                                          # < [IN][OUT]  \~chinese 输出的BMP图片缓存    \~english Output BMP image buffer
    ('nBmpBufSize', c_uint),                                                # < [IN]  \~chinese 输出的缓冲区大小          \~english Output buffer size
    ('nBmpBufLen', c_uint),                                                 # < [OUT] \~chinese 输出的BMP图片长度         \~english Output BMP picture length

    ('enCfaMethod', MV_FG_CFA_METHOD),                                      # < [IN]  \~chinese 插值方法       \~english CFA method

    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved

]
MV_FG_SAVE_BITMAP_INFO = _MV_FG_SAVE_BITMAP_INFO_

# < \~chinese 保存JPEG图像信息            \~english Save JPEG image information
class _MV_FG_SAVE_JPEG_INFO_(Structure):
    pass
_MV_FG_SAVE_JPEG_INFO_._fields_ = [
    ('stInputImageInfo', MV_FG_INPUT_IMAGE_INFO),                             # < [IN]  \~chinese 输入图片信息     \~english Input image information
    ('pJpgBuf', POINTER(c_ubyte)),                                            # < [IN][OUT]  \~chinese 输出的JPEG图片缓存   \~english Output JPEG image buffer
    ('nJpgBufSize', c_uint),                                                  # < [IN]  \~chinese 输出的缓冲区大小          \~english Output buffer size
    ('nJpgBufLen', c_uint),                                                   # < [OUT] \~chinese 输出的JPEG图片长度        \~english Output JPEG picture length

    ('nJpgQuality', c_uint),                                                  # < [IN]  \~chinese 编码质量, (0-100]      \~english Encoding quality, (0-100]
    ('enCfaMethod', MV_FG_CFA_METHOD),                                        # < [IN]  \~chinese 插值方法               \~english CFA method

    ('nReserved', c_uint * 4),                                                # < \~chinese 保留字段     \~english Reserved
]
MV_FG_SAVE_JPEG_INFO =_MV_FG_SAVE_JPEG_INFO_

# < \~chinese 保存Tiff图像信息            \~english Save Tiff image information
class _MV_FG_SAVE_TIFF_TO_FILE_INFO_(Structure):
    pass
_MV_FG_SAVE_TIFF_TO_FILE_INFO_._fields_ = [
    ('stInputImageInfo', MV_FG_INPUT_IMAGE_INFO),                             # < [IN]  \~chinese 输入图片信息     \~english Input image information
    ('pcImagePath', POINTER(c_ubyte)),                                         # < [IN]  \~chinese 输入文件路径     \~english Input image path
    ('fXResolution', c_float),                                                # < [IN]  \~chinese 水平分辨率       \~english Horizontal resolution
    ('fYResolution', c_float),                                                # < [IN]  \~chinese 垂直分辨率       \~english Vertical resolution
    ('enResolutionUnit', MV_FG_RESOLUTION_UNIT),                              # < [IN]  \~chinese 分辨率单位       \~english Resolution unit
    ('enCfaMethod', MV_FG_CFA_METHOD),                                        # < [IN]  \~chinese 插值方法               \~english CFA method

    ('nReserved', c_uint * 4),                                                # < \~chinese 保留字段     \~english Reserved

]
MV_FG_SAVE_TIFF_TO_FILE_INFO = _MV_FG_SAVE_TIFF_TO_FILE_INFO_

# < \~chinese 保存Png图像信息            \~english Save Png image information
class _MV_FG_SAVE_PNG_TO_FILE_INFO_(Structure):
    pass
_MV_FG_SAVE_PNG_TO_FILE_INFO_._fields_ = [
    ('stInputImageInfo',  MV_FG_INPUT_IMAGE_INFO),                           # < [IN]  \~chinese 输入图片信息     \~english Input image information
    ('pcImagePath', POINTER(c_ubyte)),                                        # < [IN]  \~chinese 输入文件路径     \~english Input image path
    ('nPngQuality', c_uint),                                                 # < [IN]  \~chinese 编码质量, [0-9]      \~english Encoding quality, [0-9]
    ('enCfaMethod', MV_FG_CFA_METHOD),                                       # < [IN]  \~chinese 插值方法               \~english CFA method

    ('nReserved', c_uint * 4),                                               # < \~chinese 保留字段     \~english Reserved

]
MV_FG_SAVE_PNG_TO_FILE_INFO = _MV_FG_SAVE_PNG_TO_FILE_INFO_

# < \~chinese Gamma信息, 设置Gamma曲线矫正时，需输入有效的矫正曲线   \~english Gamma information, setting gamma curve correction, input a valid correction curve
class _MV_FG_GAMMA_INFO_(Structure):
    pass
_MV_FG_GAMMA_INFO_._fields_ = [
    ('enGammaType', MV_FG_GAMMA_TYPE),                                        # < [IN]  \~chinese Gamma类型              \~english Gamma type
    ('fGammaValue', c_float),                                                 # < [IN]  \~chinese Gamma值[0.1,4.0]       \~english Gamma value[0.1,4.0]
    ('pGammaCurveBuf', POINTER(c_ubyte)),                                     # < [IN]  \~chinese Gamma曲线缓存          \~english Gamma curve buffer
    ('nGammaCurveBufLen', c_uint),                                            # < [IN]  \~chinese Gamma曲线长度          \~english Gamma curve buffer size

    ('nReserved', c_uint * 4),                                                # < \~chinese 保留字段     \~english Reserved
]
MV_FG_GAMMA_INFO = _MV_FG_GAMMA_INFO_

# < \~chinese CCM信息         \~english CCM information
class _MV_FG_CCM_INFO_(Structure):
    pass
_MV_FG_CCM_INFO_._fields_ = [
    ('bCCMEnable', c_bool),                                                   # < [IN]  \~chinese 是否启用CCM            \~english CCM enable

    ('nCCMat', c_int * 9),                                                    # < [IN]  \~chinese CCM矩阵(-65536~65536)  \~english Color correction matrix(-65536~65536)
    ('nCCMScale', c_uint),                                                    # < [IN]  \~chinese 量化系数（2的整数幂,最大65536）    \~english Quantitative scale(Integer power of 2, <= 65536)

    ('nReserved', c_uint * 4),                                                # < \~chinese 保留字段     \~english Reserved

]
MV_FG_CCM_INFO = _MV_FG_CCM_INFO_

# < \~chinese 像素格式转换信息          \~english Pixel format conversion information
class _MV_FG_CONVERT_PIXEL_INFO_(Structure):
    pass
_MV_FG_CONVERT_PIXEL_INFO_._fields_ = [
    ('stInputImageInfo', MV_FG_INPUT_IMAGE_INFO),                            # < [IN]  \~chinese 输入图片信息     \~english Input image information
    ('stOutputImageInfo', MV_FG_OUTPUT_IMAGE_INFO),                          # < [IN][OUT]  \~chinese 输出图片信息    \~english Output image information

    ('enCfaMethod', MV_FG_CFA_METHOD),                                       # < [IN]  \~chinese 插值方法   \~english CFA method
    ('bFilterEnable', c_bool),                                               # < [IN]  \~chinese 平滑使能   \~english Filter enable
    ('stGammaInfo', MV_FG_GAMMA_INFO),                                       # < [IN]  \~chinese Gamma信息  \~english Gamma information
    ('stCCMInfo', MV_FG_CCM_INFO),                                           # < [IN]  \~chinese CCM信息,仅windows支持    \~english CCM information，support only windows

    ('nReserved', c_uint * 4),                                               # < \~chinese 保留字段     \~english Reserved
]
MV_FG_CONVERT_PIXEL_INFO = _MV_FG_CONVERT_PIXEL_INFO_

# < \~chinese 水印信息                  \~english  Frame-specific information
class _MV_FG_FRAME_SPEC_INFO_(Structure):
    pass
_MV_FG_FRAME_SPEC_INFO_._fields_ = [
    # < \~chinese 设备水印时标      \~english Device frame-specific time scale
    ('nSecondCount', c_uint),                                                # < [OUT] \~chinese 秒数                   \~english The Seconds
    ('nCycleCount', c_uint),                                                 # < [OUT] \~chinese 周期数                 \~english The Count of Cycle
    ('nCycleOffset', c_uint),                                                # < [OUT] \~chinese 周期偏移量             \~english The Offset of Cycle

    # < \~chinese 白平衡相关        \~english White balance
    ('nRed', c_uint),                                                        # < [OUT] \~chinese 红色                   \~english Red
    ('nGreen', c_uint),                                                      # < [OUT] \~chinese 绿色                   \~english Green
    ('nBlue', c_uint),                                                       # < [OUT] \~chinese 蓝色                   \~english Blue

    ('nFrameCounter', c_uint),                                               # < [OUT] \~chinese 总帧数                 \~english Frame Counter
    ('nTriggerIndex', c_uint),                                               # < [OUT] \~chinese 触发计数               \~english Trigger Counting

    ('nInput', c_uint),                                                      # < [OUT] \~chinese 输入                   \~english Input
    ('nOutput', c_uint),                                                     # < [OUT] \~chinese 输出                   \~english Output

    # < \~chinese ROI区域           \~english ROI Region
    ('nOffsetX', c_ushort),                                                  # < [OUT] \~chinese 水平偏移量             \~english OffsetX
    ('nOffsetY', c_ushort),                                                  # < [OUT] \~chinese 垂直偏移量             \~english OffsetY
    ('nFrameWidth', c_ushort),                                               # < [OUT] \~chinese 水印宽                 \~english The Width of Chunk
    ('nFrameHeight', c_ushort),                                              # < [OUT] \~chinese 水印高                 \~english The Height of Chunk

    ('nReserved', c_uint * 16),                                              # < \~chinese 保留字段     \~english Reserved
]
MV_FG_FRAME_SPEC_INFO = _MV_FG_FRAME_SPEC_INFO_

# < \~chinese 无损解码参数              \~english High Bandwidth decode structure
class _MV_FG_HB_DECODE_PARAM_T_(Structure):
    pass
_MV_FG_HB_DECODE_PARAM_T_._fields_ = [
    ('pSrcBuf', POINTER(c_ubyte)),                                          # < [IN]  \~chinese 输入数据缓存           \~english Input data buffer
    ('nSrcLen', c_uint),                                                    # < [IN]  \~chinese 输入数据大小           \~english Input data size

    ('stOutputImageInfo', MV_FG_OUTPUT_IMAGE_INFO),                         # < [IN][OUT]  \~chinese 输出图片信息    \~english Output image information
    ('stFrameSpecInfo', MV_FG_FRAME_SPEC_INFO),                             # < [OUT] \~chinese 水印信息               \~english Frame Spec Info

    ('nReserved', c_uint * 8),                                              # < \~chinese 保留字段     \~english Reserved
]
MV_FG_HB_DECODE_PARAM = _MV_FG_HB_DECODE_PARAM_T_

# < \~chinese JPEG解码参数             \~english JPEG decoding structure
class _MV_FG_DECODE_JPEG_PARAM_T_(Structure):
    pass
_MV_FG_DECODE_JPEG_PARAM_T_._fields_ = [
    ('pSrcBuf', POINTER(c_ubyte)),                                          # < [IN]  \~chinese 输入数据缓存           \~english Input data buffer
    ('nSrcLen', c_uint),                                                    # < [IN]  \~chinese 输入数据大小           \~english Input data size

    ('stOutputImageInfo', MV_FG_OUTPUT_IMAGE_INFO),                         # < [IN][OUT]  \~chinese 输出图片信息    \~english Output image information

    ('nReserved', c_uint * 8),                                              # < \~chinese 保留字段     \~english Reserved
]
MV_FG_DECODE_JPEG_PARAM = _MV_FG_DECODE_JPEG_PARAM_T_

# < \~chinese 重组信息          \~english Reconstruct information
class _MV_FG_RECONSTRUCT_INFO_(Structure):
    pass
_MV_FG_RECONSTRUCT_INFO_._fields_ = [
    ('stInputImageInfo', MV_FG_INPUT_IMAGE_INFO),                           # < [IN]  \~chinese 输入图片信息   \~english Input image information
    ('stOutputImageInfo', MV_FG_OUTPUT_IMAGE_INFO * MV_FG_MAX_SPLIT_NUM),   # < [IN][OUT]  \~chinese 输出图片信息    \~english Output image information
    ('enReconstructMode', MV_FG_RECONSTRUCT_MODE),                          # < [IN]  \~chinese 重组方式       \~english Reconstruct mode

    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved
]
MV_FG_RECONSTRUCT_INFO = _MV_FG_RECONSTRUCT_INFO_

# < \~chinese Int类型信息           \~english Integer type information
class _MV_FG_INTVALUE_(Structure):
    pass
_MV_FG_INTVALUE_._fields_ = [
    ('nCurValue', c_int64),                                                 # < [OUT] \~chinese 当前值    \~english Current value
    ('nMax', c_int64),                                                      # < [OUT] \~chinese 最大值    \~english Maximum value
    ('nMin', c_int64),                                                      # < [OUT] \~chinese 最小值    \~english Minimum value
    ('nInc', c_int64),                                                      # < [OUT] \~chinese 步长      \~english Increment

    ('nReserved', c_uint * 16),                                             # < \~chinese 保留字段     \~english Reserved
]
MV_FG_INTVALUE = _MV_FG_INTVALUE_

# < \~chinese 枚举类型信息            \~english Enumeration type information
class _MV_FG_ENUMVALUE_(Structure):
    pass
_MV_FG_ENUMVALUE_._fields_ = [
    ('nCurValue', c_uint),                                                  # < [OUT] \~chinese 当前值             \~english Current value
    ('strCurSymbolic', c_byte * MV_FG_MAX_SYMBOLIC_STRLEN),                 # < [OUT] \~chinese 当前值的符号名称   \~english The symbolic name of the current value
    ('nSupportedNum', c_uint),                                              # < [OUT] \~chinese 支持的枚举类型个数      \~english Number of supported enumeration types
    ('nSupportValue', c_uint * MV_FG_MAX_SYMBOLIC_NUM),                     # < [OUT] \~chinese 支持的枚举类型的值      \~english Values of supported enumeration types
    ('strSymbolic', (c_uint * MV_FG_MAX_SYMBOLIC_NUM) * MV_FG_MAX_SYMBOLIC_STRLEN),                                   # < [OUT] \~chinese 支持的枚举类型的符号名称    \~english Symbolic names of supported enumeration types

    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved
]
MV_FG_ENUMVALUE = _MV_FG_ENUMVALUE_

# < \~chinese Float类型信息         \~english Float type information
class _MV_FG_FLOATVALUE_(Structure):
    pass
_MV_FG_FLOATVALUE_._fields_ = [
    ('fCurValue', c_float),                                                 # < [OUT] \~chinese 当前值    \~english Current value
    ('fMax', c_float),                                                      # < [OUT] \~chinese 最大值    \~english Maximum value
    ('fMin', c_float),                                                      # < [OUT] \~chinese 最小值    \~english Minimum value

    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved
]
MV_FG_FLOATVALUE = _MV_FG_FLOATVALUE_

# < \~chinese String类型信息            \~english String type information
class _MV_FG_STRINGVALUE_(Structure):
    pass
_MV_FG_STRINGVALUE_._fields_ = [
    ('strCurValue', c_byte * 256),                                          # < [OUT] \~chinese 当前值     \~english Current value
    ('nMaxLength', c_int64),                                                # < [OUT] \~chinese 最大长度   \~english Maximum length

    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved
]
MV_FG_STRINGVALUE = _MV_FG_STRINGVALUE_

# < \~chinese 文件存取            \~english File Access
class _MV_FG_FILE_ACCESS_(Structure):
    pass
_MV_FG_FILE_ACCESS_._fields_ = [
    ('pUserFileName', POINTER(c_byte)),                                          # < [IN]  \~chinese 用户文件名             \~english User file name
    ('pDevFileName', POINTER(c_byte)),                                           # < [IN]  \~chinese 设备文件名             \~english Device file name

    ('nReserved', c_uint * 32),                                                  # < \~chinese 保留字段     \~english Reserved
]
MV_FG_FILE_ACCESS = _MV_FG_FILE_ACCESS_

# < \~chinese 事件信息          \~english Event information
class _MV_FG_EVENT_INFO_(Structure):
    pass
_MV_FG_EVENT_INFO_._fields_ = [
    ('EventName', c_byte * MV_FG_MAX_EVENT_NAME_SIZE),                      # < [OUT] \~chinese 事件名称   \~english Event name
    ('nEventID', c_uint),                                                   # < [OUT] \~chinese 事件号     \~english Event ID
    ('nBlockID', c_uint64),                                                 # < [OUT] \~chinese 帧号，流事件有效   \~english Frame ID, stream event valid
    ('nTimestamp', c_uint64),                                               # < [OUT] \~chinese 时间戳     \~english Timestamp
    ('pEventData', c_void_p),                                               # < [OUT] \~chinese 事件数据，内部缓存，需要及时进行数据处理   \~english Event data, internal buffer, need to be processed in time
    ('nEventDataSize', c_uint),                                             # < [OUT] \~chinese 事件数据长度     \~english Event data szie

    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved
]
MV_FG_EVENT_INFO = _MV_FG_EVENT_INFO_

# < \~chinese 自定义点                    \~english Point defined
class _MVFG_POINTF(Structure):
    pass
_MVFG_POINTF._fields_ = [
    ('fX', c_float),                                                        # <  \~chinese 该点距离图像左边缘距离，根据图像的相对位置，范围为[0.0 , 1.0]   \~english Distance From Left，Range[0.0, 1.0]
    ('fY', c_float),                                                        # <  \~chinese 该点距离图像上边缘距离，根据图像的相对位置，范围为[0.0 , 1.0]   \~english Distance From Top，Range[0.0, 1.0]

    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved
]
MVFG_POINTF = _MVFG_POINTF

# < \~chinese 辅助线颜色                \~english Color of Auxiliary Line
class _MVFG_COLORF(Structure):
    pass
_MVFG_COLORF._fields_ = [
    ('fR', c_float),                                                        # <  \~chinese 红色，根据像素颜色的相对深度，范围为[0.0 , 1.0]，代表着[0, 255]的颜色深度   \~english Red，Range[0.0, 1.0]
    ('fG', c_float),                                                        # <  \~chinese 绿色，根据像素颜色的相对深度，范围为[0.0 , 1.0]，代表着[0, 255]的颜色深度   \~english Green，Range[0.0, 1.0]
    ('fB', c_float),                                                        # <  \~chinese 蓝色，根据像素颜色的相对深度，范围为[0.0 , 1.0]，代表着[0, 255]的颜色深度   \~english Blue，Range[0.0, 1.0]
    ('fAlpha', c_float),                                                    # <  \~chinese 透明度，根据像素颜色的相对透明度，范围为[0.0 , 1.0] (此参数功能暂不支持)    \~english Alpha，Range[0.0, 1.0](Not Support)

    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved
]
MVFG_COLORF = _MVFG_COLORF

# < \~chinese 矩形框区域信息
class _MVFG_RECT_INFO(Structure):
    pass
_MVFG_RECT_INFO._fields_ = [
    ('fTop', c_float),                                                      # <  \~chinese 矩形上边缘距离图像上边缘的距离，根据图像的相对位置，范围为[0.0 , 1.0]   \~english Distance From Top，Range[0, 1.0]
    ('fBottom', c_float),                                                   # <  \~chinese 矩形下边缘距离图像下边缘的距离，根据图像的相对位置，范围为[0.0 , 1.0]   \~english Distance From Bottom，Range[0, 1.0]
    ('fLeft', c_float),                                                     # <  \~chinese 矩形左边缘距离图像左边缘的距离，根据图像的相对位置，范围为[0.0 , 1.0]   \~english Distance From Left，Range[0, 1.0]
    ('fRight', c_float),                                                    # <  \~chinese 矩形右边缘距离图像右边缘的距离，根据图像的相对位置，范围为[0.0 , 1.0]   \~english Distance From Right，Range[0, 1.0]

    ('stColor', MVFG_COLORF),                                               # <  \~chinese 辅助线颜色                      \~english Color of Auxiliary Line
    ('nLineWidth', c_uint),                                                 # <  \~chinese 辅助线宽度，宽度只能是1或2      \~english Width of Auxiliary Line, width is 1 or 2
    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved
]
MVFG_RECT_INFO = _MVFG_RECT_INFO

# < \~chinese 圆形框区域信息            \~english Circle Area Info
class _MVFG_CIRCLE_INFO(Structure):
    pass
_MVFG_CIRCLE_INFO._fields_ = [
    ('stCenterPoint', MVFG_POINTF),                                         # <  \~chinese 圆心信息                        \~english Circle Point Info

    ('fR1', c_float),                                                       # <  \~chinese 宽向半径，根据图像的相对位置[0, 1.0]，半径与圆心的位置有关，需保证画出的圆在显示框范围之内，否则报错  \~english Windth Radius, Range[0, 1.0]
    ('fR2', c_float),                                                       # <  \~chinese 高向半径，根据图像的相对位置[0, 1.0]，半径与圆心的位置有关，需保证画出的圆在显示框范围之内，否则报错  \~english Height Radius, Range[0, 1.0]

    ('stColor', MVFG_COLORF),                                               # <  \~chinese 辅助线颜色                      \~english Color of Auxiliary Line
    ('nLineWidth', c_uint),                                                 # <  \~chinese 辅助线宽度，宽度只能是1或2      \~english Width of Auxiliary Line, width is 1 or 2
    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved

]
MVFG_CIRCLE_INFO = _MVFG_CIRCLE_INFO

# < \~chinese 线条辅助线信息    \~english Linear Auxiliary Line Info
class _MVFG_LINES_INFO(Structure):
    pass
_MVFG_LINES_INFO._fields_ = [
    ('stStartPoint', MVFG_POINTF),                                          # <  \~chinese 线条辅助线的起始点坐标          \~english The Start Point of Auxiliary Line
    ('stEndPoint', MVFG_POINTF),                                            # <  \~chinese 线条辅助线的终点坐标            \~english The End Point of Auxiliary Line

    ('stColor', MVFG_COLORF),                                               # <  \~chinese 辅助线颜色                      \~english Color of Auxiliary Line
    ('nLineWidth', c_uint),                                                 # <  \~chinese 辅助线宽度，宽度只能是1或2      \~english Width of Auxiliary Line, width is 1 or 2
    ('nReserved', c_uint * 4),                                              # < \~chinese 保留字段     \~english Reserved
]
MVFG_LINES_INFO = _MVFG_LINES_INFO