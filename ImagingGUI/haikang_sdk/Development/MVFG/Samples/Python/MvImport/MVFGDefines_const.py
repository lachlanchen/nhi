#!/usr/bin/env python
# -*- coding: utf-8 -*-

# < \~ch///< \~chinese 采集卡类型         \~english Interface type
MV_FG_GEV_INTERFACE                        = 0x00000001       # < \~chinese 网口采集卡        \~english GEV interface
MV_FG_U3V_INTERFACE                        = 0x00000002       # < \~chinese U3V采集卡         \~english U3V interface
MV_FG_CAMERALINK_INTERFACE                 = 0x00000004       # < \~chinese CameraLink采集卡  \~english CameraLink interface
MV_FG_CXP_INTERFACE                        = 0x00000008       # < \~chinese CXP采集卡         \~english CXP interface
MV_FG_XoF_INTERFACE                        = 0x00000010       # < \~chinese XoF采集卡         \~english XoF interface


MV_FG_MAX_IF_INFO_SIZE                     = 64               # < \~chinese 卡信息的字符串最大长度      \~english Maximum string length of interface information

# < \~chinese 采集卡权限         \~english Interface access mode
MV_FG_ACCESS_UNKNOWN                       = 0x0              # <\~chinese 权限未定义    \~english Access mode not defined
MV_FG_ACCESS_READONLY                      = 0x1              # < \~chinese 只读权限，无法设置或者获取节点值  \~english Read only, unable to set or get node value
MV_FG_ACCESS_CONTROL                       = 0x2              # < \~chinese 控制权限      \~english Control mode


MV_FG_MAX_DEV_INFO_SIZE                    = 64               # < \~chinese 设备信息的最大长度        \~english Maximum length of device information

MV_FG_GEV_IFCONFIG_LLA_BIT                 = 0x00000004       # < \~chinese GEV设备LLA使能标志位    \~english LLA enable flag bit of GEV device
MV_FG_GEV_IFCONFIG_DHCP_BIT                = 0x00000002       # < \~chinese GEV设备DHCP使能标志位   \~english DHCP enable flag bit of GEV device
MV_FG_GEV_IFCONFIG_PERSISTENT_BIT          = 0x00000001       # < \~chinese GEV设备静态IP使能标志位        \~english Persistent IP enable flag bit of GEV device
MV_FG_GEV_IFCONFIG_PR_BIT                  = 0x80000000       # < \~chinese GEV设备能否处理PAUSE帧标志位   \~english Can the GEV device handle the pause frame flag bit
MV_FG_GEV_IFCONFIG_PG_BIT                  = 0x40000000       # < \~chinese GEV设备能否生成PAUSE帧标志位   \~english Can the GEV device generate pause frame flag bit

# < \~chinese 设备类型         \~english Device type
MV_FG_GEV_DEVICE                           = 0x00000001       # < \~chinese 网口设备        \~english GEV device
MV_FG_U3V_DEVICE                           = 0x00000002       # < \~chinese U3V设备         \~english U3V device
MV_FG_CAMERALINK_DEVICE                    = 0x00000003       # < \~chinese CameraLink设备   \~english CameraLink device
MV_FG_CXP_DEVICE                           = 0x00000004       # < \~chinese CXP设备         \~english CXP device
MV_FG_XoF_DEVICE                           = 0x00000005       # < \~chinese XoF设备         \~english XoF device

# < \~chinese 缓存队列类型           \~english Buffer queue type
MV_FG_BUFFER_QUEUE_INPUT_TO_OUTPUT          = 0              # < \~chinese 将输入队列的BUFFER放到输出队列   \~english Put BUFFER of input queue into output queue
MV_FG_BUFFER_QUEUE_OUTPUT_DISCARD           = 1              # < \~chinese 放弃输出队列的BUFFER              \~english Discard BUFFER of output queue
MV_FG_BUFFER_QUEUE_ALL_TO_INPUT             = 2              # < \~chinese 将所有的BUFFER(包括输出队列)放到输入队列  \~english Put all BUFFER (including output queue) into input queue
MV_FG_BUFFER_QUEUE_UNQUEUED_TO_INPUT        = 3              # < \~chinese 将未使用的BUFFER放到输入队列     \~english Put unused BUFFER into input queue
MV_FG_BUFFER_QUEUE_ALL_DISCARD              = 4              # < \~chinese 放弃输入和输出队列中的BUFFER     \~english Discard BUFFER in input and output queues


# < \~chinese 像素格式定义           \~english Pixel format definition
MV_FG_PIXEL_MONO                           = 0x01000000       # < \~chinese Mono格式        \~english Mono format
MV_FG_PIXEL_COLOR                          = 0x02000000       # < \~chinese 彩色格式        \~english Color format
MV_FG_PIXEL_CUSTOM                         = 0x80000000       # < \~chinese 自定义          \~english custom

MV_FG_PIXEL_BIT_COUNT_8                    = eval("8 << 16")
MV_FG_PIXEL_BIT_COUNT_12                   = eval("12 << 16")
MV_FG_PIXEL_BIT_COUNT_16                   = eval("16 << 16")
MV_FG_PIXEL_BIT_COUNT_24                   = eval("24 << 16")
MV_FG_PIXEL_BIT_COUNT_32                   = eval("32 << 16")
MV_FG_PIXEL_BIT_COUNT_48                   = eval("48 << 16")

MV_FG_PIXEL_TYPE_Undefined                  = 0xFFFFFFFF     # < \~chinese 未定义的格式       \~english Undefined format

# <  \~chinese Mono格式           \~english Mono format
MV_FG_PIXEL_TYPE_Mono8                      = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_8 | 0x0001")   # < \~chinese Mono8            \~english Mono8
MV_FG_PIXEL_TYPE_Mono10                     = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x0003")  # < \~chinese Mono10           \~english Mono10
MV_FG_PIXEL_TYPE_Mono10_Packed              = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_12 | 0x0004")  # < \~chinese Mono10_Packed    \~english Mono10_Packed
MV_FG_PIXEL_TYPE_Mono12                     = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x0005")  # < \~chinese Mono12           \~english Mono12
MV_FG_PIXEL_TYPE_Mono12_Packed              = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_12 | 0x0006")  # < \~chinese Mono12_Packed    \~english Mono12_Packed
MV_FG_PIXEL_TYPE_Mono16                     = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x0007")  # < \~chinese Mono16           \~english Mono16

# < \~chinese Bayer格式          \~english Bayer format
MV_FG_PIXEL_TYPE_BayerGR8                   = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_8 | 0x0008")   # < \~chinese BayerGR8         \~english BayerGR8
MV_FG_PIXEL_TYPE_BayerRG8                   = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_8 | 0x0009")   # < \~chinese BayerRG8         \~english BayerRG8
MV_FG_PIXEL_TYPE_BayerGB8                   = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_8 | 0x000A")   # < \~chinese BayerGB8         \~english BayerGB8
MV_FG_PIXEL_TYPE_BayerBG8                   = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_8 | 0x000B")   # < \~chinese BayerBG8         \~english BayerBG8
MV_FG_PIXEL_TYPE_BayerGR10                  = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x000C")  # < \~chinese BayerGR10        \~english BayerGR10
MV_FG_PIXEL_TYPE_BayerRG10                  = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x000D")  # < \~chinese BayerRG10        \~english BayerRG10
MV_FG_PIXEL_TYPE_BayerGB10                  = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x000E")  # < \~chinese BayerGB10        \~english BayerGB10
MV_FG_PIXEL_TYPE_BayerBG10                  = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x000F")  #< \~chinese BayerBG10        \~english BayerBG10
MV_FG_PIXEL_TYPE_BayerGR12                  = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x0010")  # < \~chinese BayerGR12        \~english BayerGR12
MV_FG_PIXEL_TYPE_BayerRG12                  = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x0011")  # < \~chinese BayerRG12        \~english BayerRG12
MV_FG_PIXEL_TYPE_BayerGB12                  = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x0012")  # < \~chinese BayerGB12        \~english BayerGB12
MV_FG_PIXEL_TYPE_BayerBG12                  = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x0013")  # < \~chinese BayerBG12        \~english BayerBG12
MV_FG_PIXEL_TYPE_BayerGR10_Packed           = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_12 | 0x0026")  # < \~chinese BayerGR10_Packed \~english BayerGR10_Packed
MV_FG_PIXEL_TYPE_BayerRG10_Packed           = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_12 | 0x0027")  # < \~chinese BayerRG10_Packed \~english BayerRG10_Packed
MV_FG_PIXEL_TYPE_BayerGB10_Packed           = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_12 | 0x0028")  # < \~chinese BayerGB10_Packed \~english BayerGB10_Packed
MV_FG_PIXEL_TYPE_BayerBG10_Packed           = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_12 | 0x0029")  # < \~chinese BayerBG10_Packed \~english BayerBG10_Packed
MV_FG_PIXEL_TYPE_BayerGR12_Packed           = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_12 | 0x002A")  # < \~chinese BayerGR12_Packed \~english BayerGR12_Packed
MV_FG_PIXEL_TYPE_BayerRG12_Packed           = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_12 | 0x002B")  # < \~chinese BayerRG12_Packed \~english BayerRG12_Packed
MV_FG_PIXEL_TYPE_BayerGB12_Packed           = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_12 | 0x002C")  # < \~chinese BayerGB12_Packed \~english BayerGB12_Packed
MV_FG_PIXEL_TYPE_BayerBG12_Packed           = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_12 | 0x002D")  # < \~chinese BayerBG12_Packed \~english BayerBG12_Packed
MV_FG_PIXEL_TYPE_BayerGR16                  = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x002E")  # < \~chinese BayerGR16        \~english BayerGR16
MV_FG_PIXEL_TYPE_BayerRG16                  = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x002F")  # < \~chinese BayerRG16        \~english BayerRG16
MV_FG_PIXEL_TYPE_BayerGB16                  = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x0030")  # < \~chinese BayerGB16        \~english BayerGB16
MV_FG_PIXEL_TYPE_BayerBG16                  = eval("MV_FG_PIXEL_MONO | MV_FG_PIXEL_BIT_COUNT_16 | 0x0031")  # < \~chinese BayerBG16        \~english BayerBG16

# < \~chinese RGB格式            \~english RGB format
MV_FG_PIXEL_TYPE_RGB8_Packed                = eval("MV_FG_PIXEL_COLOR | MV_FG_PIXEL_BIT_COUNT_24 | 0x0014")  # < \~chinese RGB8_Packed     \~english RGB8_Packed
MV_FG_PIXEL_TYPE_BGR8_Packed                = eval("MV_FG_PIXEL_COLOR | MV_FG_PIXEL_BIT_COUNT_24 | 0x0015")  # < \~chinese BGR8_Packed     \~english BGR8_Packed
MV_FG_PIXEL_TYPE_RGBA8_Packed               = eval("MV_FG_PIXEL_COLOR | MV_FG_PIXEL_BIT_COUNT_32 | 0x0016")  # < \~chinese RGBA8_Packed    \~english RGBA8_Packed
MV_FG_PIXEL_TYPE_BGRA8_Packed               = eval("MV_FG_PIXEL_COLOR | MV_FG_PIXEL_BIT_COUNT_32 | 0x0017")  # < \~chinese BGRA8_Packed    \~english BGRA8_Packed
MV_FG_PIXEL_TYPE_RGB16_Packed               = eval("MV_FG_PIXEL_COLOR | MV_FG_PIXEL_BIT_COUNT_48 | 0x0033")  # < \~chinese RGB16_Packed    \~english RGB16_Packed
# < \~chinese YUV格式            \~english YUV format
MV_FG_PIXEL_TYPE_YUV422_Packed              = eval("MV_FG_PIXEL_COLOR | MV_FG_PIXEL_BIT_COUNT_16 | 0x001F")  # < \~chinese YUV422_Packed       \~english YUV422_Packed
MV_FG_PIXEL_TYPE_YUV422_YUYV_Packed         = eval("MV_FG_PIXEL_COLOR | MV_FG_PIXEL_BIT_COUNT_16 | 0x0032")  # < \~chinese YUV422_YUYV_Packed  \~english YUV422_YUYV_Packed



# < \~chinese 插值方法          \~english CFA method
MV_FG_CFA_METHOD_QUICK              = 0                        # < \~chinese 快速       \~english Quick
MV_FG_CFA_METHOD_BALANCE            = 1                        # < \~chinese 均衡       \~english Balance
MV_FG_CFA_METHOD_OPTIMAL            = 2                        # < \~chinese 最优       \~english Optimal

MV_FG_Resolution_Unit_None       = 1           # 无单位
MV_FG_Resolution_Unit_Inch       = 2           # 英寸
MV_FG_Resolution_Unit_CENTIMETER = 3           # 厘米

# < \~chinese Gamma类型           \~english Gamma type
MV_FG_GAMMA_TYPE_NONE               = 0                        # < \~chinese 不启用                       \~english Disable
MV_FG_GAMMA_TYPE_VALUE              = 1                        # < \~chinese Gamma值                      \~english Gamma value
MV_FG_GAMMA_TYPE_USER_CURVE         = 2                        # < \~chinese Gamma曲线                    \~english Gamma curve
                                                               # < \~chinese 输出图像数据为8位时          \~english Output image data is 8 bit
                                                               # < \~chinese      曲线长度：256*sizeof(unsigned char)      \~english 8bit,length:256*sizeof(unsigned char)
                                                               # < \~chinese 输出图像数据为16位时，曲线数据根据输入图像有所不同      \~english Output image data is 16 bit
                                                               # < \~chinese      源图像格式10位, 曲线长度：1024*sizeof(unsigned short)   \~english Input image 10bit, curve length:1024*sizeof(unsigned short)
                                                               # < \~chinese      源图像格式12位, 曲线长度：4096*sizeof(unsigned short)   \~english Input image 12bit, curve length:4096*sizeof(unsigned short)
                                                               # < \~chinese      源图像格式16位, 曲线长度：65536*sizeof(unsigned short)  \~english Input image 16bit, curve length:65536*sizeof(unsigned short)
MV_FG_GAMMA_TYPE_LRGB2SRGB          = 3                        # < \~chinese linear RGB to sRGB           \~english linear RGB to sRGB
MV_FG_GAMMA_TYPE_SRGB2LRGB          = 4                        # < \~chinese sRGB to linear RGB(仅色彩插值时支持，色彩校正时无效) \~english sRGB to linear RGB

# \~chinese 图像重组方式          \~english Image reconstruct mode
MV_FG_MAX_SPLIT_NUM                        = 8                # < \~chinese 最多将源图像拆分的个数  \~english The maximum number of source images to split
MV_FG_ROTATE_MODE                          = 0x1000           # < \~chinese 旋转方式       \~english Rotation mode
MV_FG_FLIP_MODE                            = 0x2000           # < \~chinese 翻转方式       \~english Flip mode
MV_FG_SPLIT_BY_LINE_MODE                   = 0x3000           # < \~chinese 按行拆分方式   \~english Split by row

# < \~chinese 重组方式          \~english Reconstruct mode
# < \~chinese 旋转和翻转只支持MV_FG_PIXEL_TYPE_Mono8、MV_FG_PIXEL_TYPE_RGB8_Packed、MV_FG_PIXEL_TYPE_BGR8_Packed三种像素格式 \~english Rotate and flip only support MV_FG_PIXEL_TYPE_Mono8, MV_FG_PIXEL_TYPE_RGB8_Packed and MV_FG_PIXEL_TYPE_BGR8_Packed pixel formats
RECONSTRUCT_MODE_ROTATE_90                  = eval("MV_FG_ROTATE_MODE | 0x0001")         # < \~chinese 旋转90度    \~english Rotate 90 degrees
RECONSTRUCT_MODE_ROTATE_180                 = eval("MV_FG_ROTATE_MODE | 0x0002")         # < \~chinese 旋转180度   \~english Rotate 180 degrees
RECONSTRUCT_MODE_ROTATE_270                 = eval("MV_FG_ROTATE_MODE | 0x0003")         # < \~chinese 旋转270度   \~english Rotate 270 degrees

RECONSTRUCT_MODE_FLIP_VERTICAL              = eval("MV_FG_FLIP_MODE | 0x0001")           # < \~chinese 垂直翻转    \~english Flip vertically
RECONSTRUCT_MODE_FLIP_HORIZONTAL            = eval("MV_FG_FLIP_MODE | 0x0002")           # < \~chinese 水平翻转    \~english Flip horizontally

# < \~chinese 按行拆分只支持对线阵相机采集的图像进行拆分  \~english Split by line only supports splitting the image collected by the line scan camera
RECONSTRUCT_MODE_SPLIT_BY_LINE_2            = eval("MV_FG_SPLIT_BY_LINE_MODE | 0x0002")  # < \~chinese 按行拆分成2张图像    \~english Split into 2 images by line
RECONSTRUCT_MODE_SPLIT_BY_LINE_3            = eval("MV_FG_SPLIT_BY_LINE_MODE | 0x0003")  # < \~chinese 按行拆分成3张图像    \~english Split into 3 images by line
RECONSTRUCT_MODE_SPLIT_BY_LINE_4            = eval("MV_FG_SPLIT_BY_LINE_MODE | 0x0004")  # < \~chinese 按行拆分成4张图像    \~english Split into 4 images by line

# < \~chinese 节点权限         \~english Node permissions
ACCESS_MODE_NI                              = 0                # < \~chinese Value不可实现   \~english Value is not realizable
ACCESS_MODE_NA                              = 1                # < \~chinese Value不可用     \~english Value not available
ACCESS_MODE_WO                              = 2                # < \~chinese Value只写       \~english Value write only
ACCESS_MODE_RO                              = 3                # < \~chinese Value只读       \~english Value read only
ACCESS_MODE_RW                              = 4                # < \~chinese Value读写       \~english Value read and write
ACCESS_MODE_UNDEFINED                       = 5                # < \~chinese Value未定义     \~english Value undefined

# < \~chinese 节点类型         \~english Node type
INTERFACE_TYPE_Value                        = 0                # < \~chinese Value        \~english Value
INTERFACE_TYPE_Base                         = 1                # < \~chinese Base         \~english Base
INTERFACE_TYPE_Integer                      = 2                # < \~chinese Integer      \~english Integer
INTERFACE_TYPE_Boolean                      = 3                # < \~chinese Boolean      \~english Boolean
INTERFACE_TYPE_Command                      = 4                # < \~chinese Command      \~english Command
INTERFACE_TYPE_Float                        = 5                # < \~chinese Float        \~english Float
INTERFACE_TYPE_String                       = 6                # < \~chinese String       \~english String
INTERFACE_TYPE_Register                     = 7                # < \~chinese Register     \~english Register
INTERFACE_TYPE_Category                     = 8                # < \~chinese Category     \~english Category
INTERFACE_TYPE_Enumeration                  = 9                # < \~chinese Enumeration  \~english Enumeration
INTERFACE_TYPE_EnumEntry                    = 10               # < \~chinese EnumEntry    \~english EnumEntry
INTERFACE_TYPE_Port                         = 11               # < \~chinese Port         \~english Port


MV_FG_MAX_SYMBOLIC_NUM                     = 64               # < \~chinese XML描述文件中的最大符号数        \~english Maximum number of symbols in XML description file
MV_FG_MAX_SYMBOLIC_STRLEN                  = 64               # < \~chinese XML描述文件的符号最大长度        \~english Maximum symbol length of XML description file

# \~chinese 波特率           \~english Baud rate
MV_FG_BAUDRATE_9600                        = 0x00000001       # < \~chinese 9600     \~english 9600
MV_FG_BAUDRATE_19200                       = 0x00000002       # < \~chinese 19200    \~english 19200
MV_FG_BAUDRATE_38400                       = 0x00000004       # < \~chinese 38400    \~english 38400
MV_FG_BAUDRATE_57600                       = 0x00000008       # < \~chinese 57600    \~english 57600
MV_FG_BAUDRATE_115200                      = 0x00000010       # < \~chinese 115200   \~english 115200
MV_FG_BAUDRATE_230400                      = 0x00000020       # < \~chinese 230400   \~english 230400
MV_FG_BAUDRATE_460800                      = 0x00000040       # < \~chinese 460800   \~english 460800
MV_FG_BAUDRATE_921600                      = 0x00000080       # < \~chinese 921600   \~english 921600
MV_FG_BAUDRATE_AUTOMAX                     = 0x40000000       # < \~chinese 最大值   \~english Maximum value

CONFIG_CMD_INT64_BAUDRATE                   = 1               # < \~chinese 波特率，整型, 范围MV_FG_BAUDRATE_9600 ~ MV_FG_BAUDRATE_AUTOMAX, 默认值MV_FG_BAUDRATE_115200   \~english Baud rate, integer, Value range:MV_FG_BAUDRATE_9600 ~ MV_FG_BAUDRATE_AUTOMAX, Default vaule:MV_FG_BAUDRATE_115200


# < \~chinese 异常信息类型        \~english Exception information type
EXCEPTION_TYPE_SYSTEM_TEMPERATURE_UPLIMIT          = 0x0080           # < \~chinese 温度上限                            \~english Temperature upper limit
EXCEPTION_TYPE_SYSTEM_TEMPERATURE_LOWLIMIT         = 0x0081           # < \~chinese 温度下限限                          \~english Temperature lower limit
EXCEPTION_TYPE_SYSTEM_DDR_INIT                     = 0x0082           # < \~chinese DDR初始化失败                       \~english DDR Initial Error

EXCEPTION_TYPE_CARD_PACKETBUF_ERR                  = 0x0180           # < \~chinese 包缓存错误                          \~english Card Packet Buffer Error
EXCEPTION_TYPE_CARD_ACKPACKETBUF_ERR               = 0x0181           # < \~chinese 响应包缓存错误                      \~english Card Ack Packet Buffer Error
EXCEPTION_TYPE_CARD_EVENT_PACKETBUF_ERR            = 0x0182           # < \~chinese 事件包缓存错误                       \~english Card Event Packet Buffer Error

EXCEPTION_TYPE_LINK0_STREAM_CRC_ERR                = 0x1080           # < \~chinese Link0 流CRC校验错误                 \~english Link0 Stream CRC Error Event
EXCEPTION_TYPE_LINK0_STREAM_PACKET_RESEND          = 0x1081           # < \~chinese Link0 流重发包                      \~english Link0 Stream Packet Resend
EXCEPTION_TYPE_LINK0_STREAM_CTRLPACKET_ERR         = 0x1082           # < \~chinese Link0 控制包错误                    \~english Link0 Control Packet Error Event
EXCEPTION_TYPE_LINK0_PRETREATBUF_ERR               = 0x1090           # < \~chinese Link0 预处理缓存错误                \~english Link0 Pretreat Buffer Error
EXCEPTION_TYPE_LINK0_CAM_ACK_RECVBUF_ERR           = 0x1091           # < \~chinese Link0 相机回包接收缓存错误          \~english Link0 Camera Ack Packet Receiver Buffer Error
EXCEPTION_TYPE_LINK0_CAM_ACK_TRANSMITBUF_ERR       = 0x1092           # < \~chinese Link0 相机回包发送缓存错误         \~english Link0 Camera Packet Transmitter Buffer Error

EXCEPTION_TYPE_LINK1_STREAM_CRC_ERR                = 0x1180           # < \~chinese Link1 流CRC校验错误                 \~english Link1 Stream CRC Error Event
EXCEPTION_TYPE_LINK1_STREAM_PACKET_RESEND          = 0x1181           # < \~chinese Link1 流重发包                      \~english Link1 Stream Packet Resend
EXCEPTION_TYPE_LINK1_STREAM_CTRLPACKET_ERR         = 0x1182           # < \~chinese Link1 控制包错误                    \~english Link1 Control Packet Error Event
EXCEPTION_TYPE_LINK1_PRETREATBUF_ERR               = 0x1190           # < \~chinese Link1 预处理缓存错误                \~english Link1 Pretreat Buffer Error
EXCEPTION_TYPE_LINK1_CAM_ACK_RECVBUF_ERR           = 0x1191           # < \~chinese Link1 相机回包接收缓存错误          \~english Link1 Camera Ack Packet Receiver Buffer Error
EXCEPTION_TYPE_LINK1_CAM_ACK_TRANSMITBUF_ERR       = 0x1192           # < \~chinese Link1 相机回包发送缓存错误          \~english Link1 Camera Packet Transmitter Buffer Error

EXCEPTION_TYPE_LINK2_STREAM_CRC_ERR                = 0x1280           # < \~chinese Link2 流CRC校验错误                 \~english Link2 Stream CRC Error Event
EXCEPTION_TYPE_LINK2_STREAM_PACKET_RESEND          = 0x1281           # < \~chinese Link2 流重发包                      \~english Link2 Stream Packet Resend
EXCEPTION_TYPE_LINK2_STREAM_CTRLPACKET_ERR         = 0x1282           # < \~chinese Link2 控制包错误                    \~english Link2 Control Packet Error Event
EXCEPTION_TYPE_LINK2_PRETREATBUF_ERR               = 0x1290           # < \~chinese Link2 预处理缓存错误                \~english Link2 Pretreat Buffer Error
EXCEPTION_TYPE_LINK2_CAM_ACK_RECVBUF_ERR           = 0x1291           # < \~chinese Link2 相机回包接收缓存错误          \~english Link2 Camera Ack Packet Receiver Buffer Error
EXCEPTION_TYPE_LINK2_CAM_ACK_TRANSMITBUF_ERR       = 0x1292           # < \~chinese Link2 相机回包发送缓存错误          \~english Link2 Camera Packet Transmitter Buffer Error

EXCEPTION_TYPE_LINK3_STREAM_CRC_ERR                = 0x1380           # < \~chinese Link3 流CRC校验错误                 \~english Link3 Stream CRC Error Event
EXCEPTION_TYPE_LINK3_STREAM_PACKET_RESEND          = 0x1381           # < \~chinese Link3 流重发包                      \~english Link3 Stream Packet Resend
EXCEPTION_TYPE_LINK3_STREAM_CTRLPACKET_ERR         = 0x1382           # < \~chinese Link3 控制包错误                    \~english Link3 Control Packet Error Event
EXCEPTION_TYPE_LINK3_PRETREATBUF_ERR               = 0x1390           # < \~chinese Link3 预处理缓存错误                \~english Link3 Pretreat Buffer Error
EXCEPTION_TYPE_LINK3_CAM_ACK_RECVBUF_ERR           = 0x1391           # < \~chinese Link3 相机回包接收缓存错误          \~english Link3 Camera Ack Packet Receiver Buffer Error
EXCEPTION_TYPE_LINK3_CAM_ACK_TRANSMITBUF_ERR       = 0x1392           # < \~chinese Link3 相机回包发送缓存错误          \~english Link3 Camera Packet Transmitter Buffer Error

EXCEPTION_TYPE_STREAM0_DROP_FRAME_IMAGE            = 0x2080           # < \~chinese STREAM0 卡端图像帧缓存丢弃          \~english STREAM0 Drop Image Frame
EXCEPTION_TYPE_STREAM0_IMAGE_DATACOUNT_ERR         = 0x2081           # < \~chinese STREAM0 接收图像（大小）计数异常    \~english STREAM0 Receive Image Data Count Error
EXCEPTION_TYPE_STREAM0_DROP_FRAME_TRIGGER          = 0x2082           # < \~chinese STREAM0 卡端帧触发丢弃              \~english STREAM0 Drop Frame Trigger
EXCEPTION_TYPE_STREAM0_QUEUEBUF_ERR                = 0x2090           # < \~chinese STREAM0 QUEUE缓存异常               \~english STREAM0 QUEUE Buffer Error
EXCEPTION_TYPE_STREAM0_WDMABUF_ERR                 = 0x2091           # < \~chinese STREAM0 WDMA缓存异常                \~english STREAM0 WDMA Buffer Error
EXCEPTION_TYPE_STREAM0_RDMABUF_ERR                 = 0x2092           # < \~chinese STREAM0 RDMA缓存异常                \~english STREAM0 RDMA Buffer Error
EXCEPTION_TYPE_STREAM0_PACKETBUF_ERR               = 0x2093           # < \~chinese STREAM0 PACKET缓存异常              \~english STREAM0 PACKET Buffer Error
EXCEPTION_TYPE_STREAM0_WDMALENGTH_ERR              = 0x2094           # < \~chinese STREAM0 WDMA长度异常                \~english STREAM0 WDMA Length Error
EXCEPTION_TYPE_STREAM0_RDMALENGTH_ERR              = 0x2095           # < \~chinese STREAM0 RDMA长度异常                \~english STREAM0 RDMA Length Error

EXCEPTION_TYPE_STREAM1_DROP_FRAME_IMAGE            = 0x2180           # < \~chinese STREAM1 卡端图像帧缓存丢弃          \~english STREAM1 Drop Image Frame
EXCEPTION_TYPE_STREAM1_IMAGE_DATACOUNT_ERR         = 0x2181           # < \~chinese STREAM1 接收图像（大小）计数异常    \~english STREAM1 Receive Image Data Count Error
EXCEPTION_TYPE_STREAM1_DROP_FRAME_TRIGGER          = 0x2182           # < \~chinese STREAM1 卡端帧触发丢弃              \~english STREAM1 Drop Frame Trigger
EXCEPTION_TYPE_STREAM1_QUEUEBUF_ERR                = 0x2190           # < \~chinese STREAM1 QUEUE缓存异常               \~english STREAM1 QUEUE Buffer Error
EXCEPTION_TYPE_STREAM1_WDMABUF_ERR                 = 0x2191           # < \~chinese STREAM1 WDMA缓存异常                \~english STREAM1 WDMA Buffer Error
EXCEPTION_TYPE_STREAM1_RDMABUF_ERR                 = 0x2192           # < \~chinese STREAM1 RDMA缓存异常                \~english STREAM1 RDMA Buffer Error
EXCEPTION_TYPE_STREAM1_PACKETBUF_ERR               = 0x2193           # < \~chinese STREAM1 PACKET缓存异常              \~english STREAM1 PACKET Buffer Error
EXCEPTION_TYPE_STREAM1_WDMALENGTH_ERR              = 0x2194           # < \~chinese STREAM1 WDMA长度异常                \~english STREAM1 WDMA Length Error
EXCEPTION_TYPE_STREAM1_RDMALENGTH_ERR              = 0x2195           # < \~chinese STREAM1 RDMA长度异常                \~english STREAM1 RDMA Length Error

EXCEPTION_TYPE_STREAM2_DROP_FRAME_IMAGE            = 0x2280           # < \~chinese STREAM2 卡端图像帧缓存丢弃          \~english STREAM2 Drop Image Frame
EXCEPTION_TYPE_STREAM2_IMAGE_DATACOUNT_ERR         = 0x2281           # < \~chinese STREAM2 接收图像（大小）计数异常    \~english STREAM2 Receive Image Data Count Error
EXCEPTION_TYPE_STREAM2_DROP_FRAME_TRIGGER          = 0x2282           # < \~chinese STREAM2 卡端帧触发丢弃              \~english STREAM2 Drop Frame Trigger
EXCEPTION_TYPE_STREAM2_QUEUEBUF_ERR                = 0x2290           # < \~chinese STREAM2 QUEUE缓存异常               \~english STREAM2 QUEUE Buffer Error
EXCEPTION_TYPE_STREAM2_WDMABUF_ERR                 = 0x2291           # < \~chinese STREAM2 WDMA缓存异常                \~english STREAM2 WDMA Buffer Error
EXCEPTION_TYPE_STREAM2_RDMABUF_ERR                 = 0x2292           # < \~chinese STREAM2 RDMA缓存异常                \~english STREAM2 RDMA Buffer Error
EXCEPTION_TYPE_STREAM2_PACKETBUF_ERR               = 0x2293           # < \~chinese STREAM2 PACKET缓存异常              \~english STREAM2 PACKET Buffer Error
EXCEPTION_TYPE_STREAM2_WDMALENGTH_ERR              = 0x2294           # < \~chinese STREAM2 WDMA长度异常                \~english STREAM2 WDMA Length Error
EXCEPTION_TYPE_STREAM2_RDMALENGTH_ERR              = 0x2295           # < \~chinese STREAM2 RDMA长度异常                \~english STREAM2 RDMA Length Error

EXCEPTION_TYPE_STREAM3_DROP_FRAME_IMAGE            = 0x2380           # < \~chinese STREAM3 卡端图像帧缓存丢弃          \~english STREAM3 Drop Image Frame
EXCEPTION_TYPE_STREAM3_IMAGE_DATACOUNT_ERR         = 0x2381           # < \~chinese STREAM3 接收图像（大小）计数异常    \~english STREAM3 Receive Image Data Count Error
EXCEPTION_TYPE_STREAM3_DROP_FRAME_TRIGGER          = 0x2382           # < \~chinese STREAM3 卡端帧触发丢弃              \~english STREAM3 Drop Frame Trigger
EXCEPTION_TYPE_STREAM3_QUEUEBUF_ERR                = 0x2390           # < \~chinese STREAM3 QUEUE缓存异常               \~english STREAM3 QUEUE Buffer Error
EXCEPTION_TYPE_STREAM3_WDMABUF_ERR                 = 0x2391           # < \~chinese STREAM3 WDMA缓存异常                \~english STREAM3 WDMA Buffer Error
EXCEPTION_TYPE_STREAM3_RDMABUF_ERR                 = 0x2392           # < \~chinese STREAM3 RDMA缓存异常                \~english STREAM3 RDMA Buffer Error
EXCEPTION_TYPE_STREAM3_PACKETBUF_ERR               = 0x2393           # < \~chinese STREAM3 PACKET缓存异常              \~english STREAM3 PACKET Buffer Error
EXCEPTION_TYPE_STREAM3_WDMALENGTH_ERR              = 0x2394           # < \~chinese STREAM3 WDMA长度异常                \~english STREAM3 WDMA Length Error
EXCEPTION_TYPE_STREAM3_RDMALENGTH_ERR              = 0x2395           # < \~chinese STREAM3 RDMA长度异常                \~english STREAM3 RDMA Length Error

EXCEPTION_TYPE_PCIE_SCHEDULEBUF_ERR                = 0x3088           # < \~chinese 调度模块缓存异常                    \~english Sched Buffer Error
EXCEPTION_TYPE_PCIE_SCHEDULE_ERR                   = 0x3089           # < \~chinese 调度结果到异常值                    \~english Sched Error

EXCEPTION_TYPE_PCIE_LINK0_RECVBUF_ERR              = 0x3180           # < \~chinese Link0 缓存Link的buffer异常           \~english Link0 Reveive Buffer Error
EXCEPTION_TYPE_PCIE_LINK0_LENGTH_ERR               = 0x3181           # < \~chinese Link0 控制包长度异常                \~english Link0 Length Erro
EXCEPTION_TYPE_PCIE_LINK0_SOFT_RECVBUF_ERR         = 0x3280           # < \~chinese Link0 缓存Link的buffer异常           \~english Link0 Soft Reveive Buffer Error
EXCEPTION_TYPE_PCIE_LINK0_SOFT_LENGTH_ERR          = 0x3281           # < \~chinese Link0 控制包长度异常                \~english Link0 Soft Length Erro

EXCEPTION_TYPE_PCIE_LINK1_RECVBUF_ERR              = 0x3188           # < \~chinese Link1 缓存Link的buffer异常           \~english Link1 Reveive Buffer Error
EXCEPTION_TYPE_PCIE_LINK1_LENGTH_ERR               = 0x3189           # < \~chinese Link1 控制包长度异常                \~english Link1 Length Erro
EXCEPTION_TYPE_PCIE_LINK1_SOFT_RECVBUF_ERR         = 0x3288           # < \~chinese Link1 缓存Link的buffer异常           \~english Link1 Soft Reveive Buffer Error
EXCEPTION_TYPE_PCIE_LINK1_SOFT_LENGTH_ERR          = 0x3289           # < \~chinese Link1 控制包长度异常                \~english Link1 Soft Length Erro

EEXCEPTION_TYPE_PCIE_LINK2_RECVBUF_ERR             = 0x3190           # < \~chinese Link2 缓存Link的buffer异常           \~english Link2 Reveive Buffer Error
EXCEPTION_TYPE_PCIE_LINK2_LENGTH_ERR               = 0x3191           # < \~chinese Link2 控制包长度异常                \~english Link2 Length Erro
EXCEPTION_TYPE_PCIE_LINK2_SOFT_RECVBUF_ERR         = 0x3290           # < \~chinese Link2 缓存Link的buffer异常           \~english Link2 Soft Reveive Buffer Error
EXCEPTION_TYPE_PCIE_LINK2_SOFT_LENGTH_ERR          = 0x3291           # < \~chinese Link2 控制包长度异常                \~english Link2 Soft Length Error

EXCEPTION_TYPE_PCIE_LINK3_RECVBUF_ERR              = 0x3198           # < \~chinese Link3 缓存Link的buffer异常           \~english Link3 Reveive Buffer Error
EXCEPTION_TYPE_PCIE_LINK3_LENGTH_ERR               = 0x3199           # < \~chinese Link3 控制包长度异常                \~english Link3 Length Erro
EXCEPTION_TYPE_PCIE_LINK3_SOFT_RECVBUF_ERR         = 0x3298           # < \~chinese Link3 缓存Link的buffer异常           \~english Link3 Soft Reveive Buffer Error
EXCEPTION_TYPE_PCIE_LINK3_SOFT_LENGTH_ERR          = 0x3299           # < \~chinese Link3 控制包长度异常                \~english Link3 Soft Length Erro

EXCEPTION_TYPE_PCIE_STREAM0_RECVBUF_ERR            = 0x3382           # < \~chinese STREAM0 缓存Stream的fifo异常         \~english STREAM0 Receive Buffer Error
EXCEPTION_TYPE_PCIE_STREAM0_LIST_ERR               = 0x3383           # < \~chinese STREAM0 链表格式错误                \~english STREAM0 List Error
EXCEPTION_TYPE_PCIE_STREAM0_SIZE_ERR               = 0x3384           # < \~chinese STREAM0 图像大小与内存不匹配        \~english STREAM0 Size Error

EXCEPTION_TYPE_PCIE_STREAM1_RECVBUF_ERR            = 0x338A           # < \~chinese STREAM1 缓存Stream的fifo异常         \~english STREAM1 Receive Buffer Error
EXCEPTION_TYPE_PCIE_STREAM1_LIST_ERR               = 0x338B           # < \~chinese STREAM1 链表格式错误                \~english STREAM1 List Error
EXCEPTION_TYPE_PCIE_STREAM1_SIZE_ERR               = 0x338C           # < \~chinese STREAM1 图像大小与内存不匹配        \~english STREAM1 Size Error

EXCEPTION_TYPE_PCIE_STREAM2_RECVBUF_ERR            = 0x3392           # < \~chinese STREAM2 缓存Stream的fifo异常         \~english STREAM2 Receive Buffer Error
EXCEPTION_TYPE_PCIE_STREAM2_LIST_ERR               = 0x3393           # < \~chinese STREAM2 链表格式错误                \~english STREAM2 List Error
EXCEPTION_TYPE_PCIE_STREAM2_SIZE_ERR               = 0x3394           # < \~chinese STREAM2 图像大小与内存不匹配        \~english STREAM2 Size Error

EXCEPTION_TYPE_PCIE_STREAM3_RECVBUF_ERR            = 0x339A           # < \~chinese STREAM3 缓存Stream的fifo异常         \~english STREAM3 Receive Buffer Error
EXCEPTION_TYPE_PCIE_STREAM3_LIST_ERR               = 0x339B           # < \~chinese STREAM3 链表格式错误                \~english STREAM3 List Error
EXCEPTION_TYPE_PCIE_STREAM3_SIZE_ERR               = 0x339C           # < \~chinese STREAM3 图像大小与内存不匹配        \~english STREAM3 Size Error

EXCEPTION_TYPE_CAMERA_DISCONNECT_ERR               = 0x10000001       # < \~chinese CAMERA 相机掉线错误                \~english CAMERA Disconnect Error


MV_FG_MAX_EVENT_NAME_SIZE                          = 128              # < \~chinese 事件名称最大长度         \~english Maximum length of event name

# \~chinese 取流策略                  \~english The strategy of Grabbing
MV_FG_GrabStrategy_OneByOne                        = 0               # < \~chinese 从旧到新一帧一帧的获取图像   \~english Grab One By One
MV_FG_GrabStrategy_LatestImagesOnly                = 1               # < \~chinese 获取列表中最新的一帧图像     \~english Grab The Latest Image
MV_FG_GrabStrategy_LatestImages                    = 2               # < \~chinese 获取列表中最新的图像         \~english Grab The Latest Images
MV_FG_GrabStrategy_UpcomingImage                   = 3               # < \~chinese 等待下一帧图像               \~english Grab The Upcoming Image
