#pragma once
#ifndef _FMC4030_DLL_H_
#define _FMC4030_DLL_H_

#ifdef FMC4030_DLL
#define FMC4030_API  __declspec(dllexport) __stdcall
#else
#define FMC4030_API  __declspec(dllimport) __stdcall
#endif

/* 最大轴数3，此数请勿修改，会造成数据错乱导致程序崩溃 */
#define MAX_AXIS                     3

/* 设备状态@machine_status.machineRunStatus */
#define MACHINE_MANUAL                     0x0001     //设备处于手动模式
#define MACHINE_AUTO                       0x0002     //设备处于运行脚本模式

/* 各轴状态@machine_status.axisStatus */                      
#define MACHINE_POWER_ON                   0x0000     //无意义
#define MACHINE_RUNNING                    0x0001     //轴正在运行
#define MACHINE_PAUSE                      0x0002     //轴暂停运行
#define MACHINE_RESUME                     0x0004     //无
#define MACHINE_STOP                       0x0008     //轴停止运行
#define MACHINE_LIMIT_N                    0x0010     //负限位触发
#define MACHINE_LIMIT_P                    0x0020	  //正限位触发
#define MACHINE_HOME_DONE                  0x0040     //轴回零完成
#define MACHINE_HOME                       0x0080     //轴回零中
#define MACHINE_AUTO_RUN                   0x0100     //无
#define MACHINE_LIMIT_N_NONE               0x0200     //负限位未触发
#define MACHINE_LIMIT_P_NONE               0x0400     //正限位未触发
#define MACHINE_HOME_NONE                  0x0800     //未回零
#define MACHINE_HOME_OVERTIME              0x1000     //回零超时

struct machine_status{
	float realPos[MAX_AXIS];                          //当前各轴实时位置
	float realSpeed[MAX_AXIS];                        //当前各轴实时速度
	unsigned int inputStatus;                         //输入状态
	unsigned int outputStatus;                        //输出状态
	unsigned int limitNStatus;                        //负限位状态
	unsigned int limitPStatus;                        //正限位状态
	unsigned int machineRunStatus;                    //设备运行状态
	unsigned int axisStatus[MAX_AXIS];                //轴状态
	unsigned int homeStatus;                          //回零状态，未使用，通过轴状态来获取回零状态
	char file[20][30];                                //控制器内程序文件，至多20个文件
	//char currentRunFile[30];                          //当前正在运行的文件
	//int currentRunLine;                               //当前运行程序所处的行号
};

struct machine_device_para{
	unsigned int id;
	unsigned int bound232;
	unsigned int bound485;
	char ip[15];
	int port;
	
	int div[MAX_AXIS];
	int lead[MAX_AXIS];
	int softLimitMax[MAX_AXIS];
	int softLimitMin[MAX_AXIS];
	int homeTime[MAX_AXIS];
};

struct machine_version{
	unsigned int firmware;
	unsigned int lib;
	unsigned int serialnumber;
};

#ifdef __cplusplus
extern "C" {
#endif

/* 设备打开与关闭相关函数 */
int FMC4030_API FMC4030_Open_Device(int id, char* ip, int port);
int FMC4030_API FMC4030_Close_Device(int id);

/* 单轴运动与轴状态检查相关函数 */
int FMC4030_API FMC4030_Jog_Single_Axis(int id, int axis, float pos, float speed, float acc, float dec, int mode);
int FMC4030_API FMC4030_Check_Axis_Is_Stop(int id, int axis);
int FMC4030_API FMC4030_Home_Single_Axis(int id, int axis, float homeSpeed, float homeAccDec, float homeFallStep, int homeDir);
int FMC4030_API FMC4030_Stop_Single_Axis(int id, int axis, int mode);
int FMC4030_API FMC4030_Get_Axis_Current_Pos(int id, int axis, float* pos);
int FMC4030_API FMC4030_Get_Axis_Current_Speed(int id, int axis, float* speed);

/* IO操作函数 */
int FMC4030_API FMC4030_Set_Output(int id, int io, int status);
int FMC4030_API FMC4030_Get_Input(int id, int io, int* status);

/* 485总线通用数据操作函数 */
int FMC4030_API FMC4030_Write_Data_To_485(int id, char* send, int length);
int FMC4030_API FMC4030_Read_Data_From_485(int id, char* recv, int* length);

/* 485总线扩展设备操作，适用于操作本司485扩展设备（单轴控制器、IO扩展板等设备） */
int FMC4030_API FMC4030_Set_FSC_Speed(int id, int slaveId, float speed);

/* Modbus协议操作函数，适用于FMC4030485总线发送Modbus协议数据访问从机设备 */
/* 01功能码：读取单个线圈值 */
int FMC4030_API FMC4030_MB01_Operation(int id, int slaveId, unsigned short int addr, char* recv, int* recvLength);

/* 03功能码：读寄存器值 */
/** 03操作
  * id：控制器连接时分配的id号
  * slaveId：485总线从机地址
  * addr：从机地址中的Modbus起始地址
  * numOfData：读取的寄存器个数
  * recv：传入数组指针，读取的内容将会保存在其中，可填0
  * recvLength：读取到的数据长度，可填0
  */
int FMC4030_API FMC4030_MB03_Operation(int id, int slaveId, unsigned short int addr, int numOfData, char* recv, int* recvLength);

/* 05功能码：写单个线圈值，1 或 0 */
/** 05操作
  * id：控制器连接时分配的id号
  * slaveId：485总线从机地址
  * addr：从机地址中的Modbus起始地址
  * val：写入的值：0xFF00 或 0x0000
  * recv：485总线从机返回的数据，传入数组指针，读取的内容将会保存在其中，可填0
  * recvLength：读取到的数据长度，可填0
  */
int FMC4030_API FMC4030_MB05_Operation(int id, int slaveId, unsigned short int addr, unsigned short int val, char* recv, int* recvLength);

/* 06功能码：写单个寄存器，0x0000 ~ 0xFFFF */
/** 06操作
  * id：控制器连接时分配的id号
  * slaveId：485总线从机地址
  * addr：从机地址中的Modbus起始地址
  * val：写入的值
  * recv：485总线从机返回的数据，传入数组指针，读取的内容将会保存在其中，可填0
  * recvLength：读取到的数据长度，可填0
  */
int FMC4030_API FMC4030_MB06_Operation(int id, int slaveId, unsigned short int addr, unsigned short int val, char* recv, int* recvLength);

/* 16功能码：写多个寄存器 */
/** 16操作
  * id：控制器连接时分配的id号
  * slaveId：485总线从机地址
  * addr：从机地址中的Modbus起始地址
  * numOfData：待写入的数据个数
  * send：写入值得数组，应与个数相匹配
  * recv：485总线从机返回的数据，传入数组指针，读取的内容将会保存在其中，可填0
  * recvLength：读取到的数据长度，可填0
  */
int FMC4030_API FMC4030_MB16_Operation(int id, int slaveId, unsigned short int addr, int numOfData, unsigned short int* send, char* recv, int* recvLength);

/* 多轴插补相关函数 */
/** 功能：以当前点为起点的两轴直线插补运动
  * id：卡号
  * axis：插补的轴号，采用十六进制的低三位表示X、Y、Z轴，可选组合有（0x03、0x05、0x06）
  * endX：终点的X坐标值
  * endY：终点的Y坐标值
  * speed：插补速度，此速度为两轴合成速度，并非各独立轴实际运行速度
  * acc：插补加速度，此加速度同样为合成加速度
  * dcc：插补减速度，此减速度同样为合成减速度
  */
int FMC4030_API FMC4030_Line_2Axis(int id, unsigned int axis, float endX, float endY, float speed, float acc, float dec);

/** 功能：以当前点为起点的三轴直线插补运动
  * id：卡号
  * endX：终点的X坐标值
  * endY：终点的Y坐标值
  * endZ：终点的Z坐标值
  * speed：插补速度，此速度为三轴合成速度，并非各独立轴实际运行速度
  * acc：插补加速度，此加速度同样为合成加速度
  * dcc：插补减速度，此减速度同样为合成减速度
  */
int FMC4030_API FMC4030_Line_3Axis(int id, unsigned int axis, float endX, float endY, float endZ, float speed, float acc, float dec);

/** 功能：以当前点为起点的两轴圆弧插补运动
  * id：卡号
  * axis：插补的轴号，采用十六进制的低三位表示X、Y、Z轴，可选组合有（0x03、0x05、0x06）
  * endX：终点的X坐标值
  * endY：终点的Y坐标值
  * centerX：圆心X坐标值
  * centerY：圆心Y坐标值
  * radius：圆半径
  * speed：插补速度，此速度为两轴合成速度，并非各独立轴实际运行速度
  * acc：插补加速度，此加速度同样为合成加速度
  * dcc：插补减速度，此减速度同样为合成减速度
  */
int FMC4030_API FMC4030_Arc_2Axis(int id, unsigned int axis, float endX, float endY, float centerX, float centerY, float radius, float speed, float acc, float dec, int dir);

int FMC4030_API FMC4030_Pause_Run(int id, unsigned int axis);
int FMC4030_API FMC4030_Resume_Run(int id, unsigned int axis);
int FMC4030_API FMC4030_Stop_Run(int id);

/* 设备状态查询相关函数 */
/** 功能：查询设备状态信息
  * id：卡号
  * machineData：传入的字符数组指针，由machine_status定义的结构体组成，调用完成后，数组将会填充数据
  */
int FMC4030_API FMC4030_Get_Machine_Status(int id, unsigned char* machineData);
int FMC4030_API FMC4030_Get_Device_Para(int id, unsigned char* devicePara);
int FMC4030_API FMC4030_Set_Device_Para(int id, unsigned char* devicePara);
int FMC4030_API FMC4030_Get_Version_Info(int id, unsigned char* version);

/* 文件操作相关函数 */
/** 功能：下载控制卡升级文件或脚本程序文件，分别为.bin  .elo,此函数在下载文件较大时会阻塞较长时间
  * id：卡号
  * filePath：文件路径
  * fileType：文件类型  1：控制卡升级程序文件  2：控制卡自动运行执行脚本文件
  */
int FMC4030_API FMC4030_Download_File(int id, char* filePath, int fileType);

/* 自动控制相关函数 */
/** 功能：启动控制卡内的程序进行自动运行
  * id：卡号
  * file：待启动的文件名，此文件名从控制卡内部获取
  */
int FMC4030_API FMC4030_Start_Auto_Run(int id, char* file);

/** 功能：停止控制卡自动运行
  * id：卡号
  */
int FMC4030_API FMC4030_Stop_Auto_Run(int id);

/** 功能：删除控制器内程序文件
  * id：卡号
  * file：待删除的文件名，此文件名从控制卡内部获取
  */
int FMC4030_API FMC4030_Delete_Script_File(int id, char* file);


#ifdef __cplusplus
}
#endif 

#endif