#!/usr/bin/env python
# -*- coding: utf-8 -*-

#typedef int MV_FG_ERROR;


MV_FG_SUCCESS                   = 0            # < \~chinese 成功，无错误             \~english Successed, no error



#SDK错误:范围0x80190001-0x801900FF

MV_FG_ERR_ERROR                  = 0x80190001    # < \~chinese 未知错误         \~english Unknown error
MV_FG_ERR_NOT_INITIALIZED        = 0x80190002    # < \~chinese 未初始化         \~english Uninitialized
MV_FG_ERR_NOT_IMPLEMENTED        = 0x80190003    # < \~chinese 未实现           \~english Not implemented
MV_FG_ERR_RESOURCE_IN_USE        = 0x80190004    # < \~chinese 资源被占用        \~english Resource occupied
MV_FG_ERR_ACCESS_DENIED          = 0x80190005    # < \~chinese 无权限           \~english No permission
MV_FG_ERR_INVALID_HANDLE         = 0x80190006    # < \~chinese 无效句柄          \~english Invalid handle
MV_FG_ERR_INVALID_ID             = 0x80190007    # < \~chinese 无效ID            \~english Invalid ID
MV_FG_ERR_NO_DATA                = 0x80190008    # < \~chinese 无数据           \~english No data
MV_FG_ERR_INVALID_PARAMETER      = 0x80190009    # < \~chinese 无效参数         \~english Invalid parameter
MV_FG_ERR_IO                     = 0x80190010    # < \~chinese IO错误           \~english IO error
MV_FG_ERR_TIMEOUT                = 0x80190011    # < \~chinese 超时             \~english Timeout
MV_FG_ERR_ABORT                  = 0x80190012    # < \~chinese 操作被中断       \~english The operation was interrupted
MV_FG_ERR_INVALID_BUFFER         = 0x80190013    # < \~chinese 无效缓存         \~english Invalid buffer
MV_FG_ERR_NOT_AVAILABLE          = 0x80190014    # < \~chinese 不可达           \~english Unreachable
MV_FG_ERR_INVALID_ADDRESS        = 0x80190015    # < \~chinese 无效地址         \~english Invalid address
MV_FG_ERR_BUFFER_TOO_SMALL       = 0x80190016    # < \~chinese 缓存太小         \~english Buffer too small
MV_FG_ERR_INVALID_INDEX          = 0x80190017    # < \~chinese 无效索引         \~english Invalid index
MV_FG_ERR_PARSING_CHUNK_DATA     = 0x80190018    # < \~chinese 解析Chunk失败    \~english Failed to parse Chunk
MV_FG_ERR_INVALID_VALUE          = 0x80190019    # < \~chinese 无效的值         \~english Invalid value
MV_FG_ERR_RESOURCE_EXHAUSTED     = 0x80190020    # < \~chinese 资源耗尽         \~english Resource exhaustion
MV_FG_ERR_OUT_OF_MEMORY          = 0x80190021    # < \~chinese 内存申请失败     \~english Applying memory failed
MV_FG_ERR_BUSY                   = 0x80190022    # < \~chinese 忙碌             \~english Be busy
MV_FG_ERR_LOADLIBRARY            = 0x80190023    # < \~chinese 动态库加载失败   \~english Dynamic library loading failed
MV_FG_ERR_CALLORDER              = 0x80190024    # < \~chinese 函数调用错误     \~english The order of function calls is wrong


#GenICam系列错误:范围0x80190100-0x801901FF

MV_FG_ERR_GC_GENERIC             = 0x80190100    # < \~chinese 通用错误         \~english General error
MV_FG_ERR_GC_ARGUMENT            = 0x80190101    # < \~chinese 参数错误         \~english Illegal parameters
MV_FG_ERR_GC_RANGE               = 0x80190102    # < \~chinese 参数范围错误     \~english The value is out of range
MV_FG_ERR_GC_PROPERTY            = 0x80190103    # < \~chinese 属性错误         \~english Property error
MV_FG_ERR_GC_RUNTIME             = 0x80190104    # < \~chinese 运行环境错误     \~english Running environment error
MV_FG_ERR_GC_LOGICAL             = 0x80190105    # < \~chinese 逻辑错误         \~english Logical error
MV_FG_ERR_GC_ACCESS              = 0x80190106    # < \~chinese 权限错误         \~english Accessing condition error
MV_FG_ERR_GC_TIMEOUT             = 0x80190107    # < \~chinese 超时             \~english Timeout
MV_FG_ERR_GC_DYNAMICCAST         = 0x80190108    # < \~chinese 转换异常         \~english Transformation exception
MV_FG_ERR_GC_UNKNOW              = 0x801901FF    # < \~chinese 未知错误         \~english Unknown error


#图像处理错误:范围0x80190200-0x801902FF

MV_FG_ERR_IMG_HANDLE             = 0x80190200    # < \~chinese 图像处理库句柄错误             \~english Handle error
MV_FG_ERR_IMG_SUPPORT            = 0x80190201    # < \~chinese 图像处理库不支持               \~english Not supported
MV_FG_ERR_IMG_PARAMETER          = 0x80190202    # < \~chinese 图像处理库参数错误             \~english Parameter error
MV_FG_ERR_IMG_OVERFLOW           = 0x80190203    # < \~chinese 图像处理库内存溢出             \~english Out of memory
MV_FG_ERR_IMG_INITIALIZED        = 0x80190204    # < \~chinese 图像处理库操作未初始化         \~english Uninitialized
MV_FG_ERR_IMG_RESOURCE           = 0x80190205    # < \~chinese 图像处理库资源申请释放失败     \~english Resource release failed
MV_FG_ERR_IMG_ENCRYPT            = 0x80190206    # < \~chinese 图像加密错误                   \~english Encryption error
MV_FG_ERR_IMG_FORMAT             = 0x80190207    # < \~chinese 图像格式不正确或者不支持       \~english Incorrect format or unsupported
MV_FG_ERR_IMG_SIZE               = 0x80190208    # < \~chinese 图像宽高不正确或者超出范围     \~english Incorrect width and height or out of range
MV_FG_ERR_IMG_STEP               = 0x80190209    # < \~chinese 图像宽高与step参数不匹配       \~english Width and height do not match step parameters
MV_FG_ERR_IMG_DATA_NULL          = 0x80190210    # < \~chinese 图像数据存储地址为空(某个分量) \~english Data storage address is empty (a component)
MV_FG_ERR_IMG_ABILITY_ARG        = 0x80190211    # < \~chinese 图像算法ABILITY存在无效参数    \~english Invalid parameter for algorithm ABILITY
MV_FG_ERR_IMG_UNKNOW             = 0x801902FF    # < \~chinese 图像处理未知错误               \~english Unknown error
