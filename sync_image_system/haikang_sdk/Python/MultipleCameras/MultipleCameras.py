# -*- coding: utf-8 -*-
import sys
import time

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QTextCursor
from CamOperation_class import CameraOperation
from MvCameraControl_class import *
from MvErrorDefine_const import *
from CameraParams_header import *
from PyUIMultipleCameras import Ui_MainWindow
import ctypes

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

# Decoding Characters
def decoding_char(c_ubyte_value):
    c_char_p_value = ctypes.cast(c_ubyte_value, ctypes.c_char_p)
    try:
        decode_str = c_char_p_value.value.decode('gbk')  # Chinese characters
    except UnicodeDecodeError:
        decode_str = str(c_char_p_value.value)
    return decode_str


if __name__ == "__main__":

    global deviceList
    deviceList = MV_CC_DEVICE_INFO_LIST()

    global cam_checked_list
    cam_checked_list = []

    global obj_cam_operation
    obj_cam_operation = []

    global win_display_handles
    win_display_handles = []

    global valid_number
    valid_number = 0

    global b_is_open
    b_is_open = False

    global b_is_grab
    b_is_grab = False

    global b_is_trigger
    b_is_trigger = False

    global b_is_software_trigger
    b_is_software_trigger = False


    # ch:初始化SDK | en: initialize SDK
    MvCamera.MV_CC_Initialize()

    # print info in ui
    def print_text(str_info):
        ui.textEdit.moveCursor(QTextCursor.Start)
        ui.textEdit.insertPlainText(str_info + "\n")

    # ch:枚举相机 | en:enum devices
    def enum_devices():
        global deviceList
        global valid_number
        deviceList = MV_CC_DEVICE_INFO_LIST()
        n_layer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE
                        | MV_GENTL_GIGE_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                        | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
        ret = MvCamera.MV_CC_EnumDevicesEx2(n_layer_type, deviceList, '', SortMethod_SerialNumber)
        if ret != 0:
            str_error = "Enum devices fail! ret = :" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", str_error, QMessageBox.Ok)
            return ret

        if deviceList.nDeviceNum == 0:
            QMessageBox.warning(mainWindow, "Info", "Find no device", QMessageBox.Ok)
            return ret
        print_text("Find %d devices!" % deviceList.nDeviceNum)

        valid_number = 0
        for i in range(0, 4):
            if (i < deviceList.nDeviceNum) is True:
                serial_number = ""
                model_name = ""
                mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
                    print("\ngige device: [%d]" % i)
                    user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName)
                    model_name = decoding_char(mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName)
                    print("device user define name: " + user_defined_name)
                    print("device model name: " + model_name)

                    nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                    nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                    nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                    nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                    print("current ip: %d.%d.%d.%d " % (nip1, nip2, nip3, nip4))

                    for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chSerialNumber:
                        if per == 0:
                            break
                        serial_number = serial_number + chr(per)

                elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                    print("\nu3v device: [%d]" % i)
                    user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName)
                    model_name = decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName)
                    print("device user define name: " + user_defined_name)
                    print("device model name: " + model_name)

                    for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                        if per == 0:
                            break
                        serial_number = serial_number + chr(per)
                    print("user serial number: " + serial_number)
                elif mvcc_dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
                    print("\nCML device: [%d]" % i)
                    user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stCMLInfo.chUserDefinedName)
                    model_name = decoding_char(mvcc_dev_info.SpecialInfo.stCMLInfo.chModelName)
                    print("device user define name: " + user_defined_name)
                    print("device model name: " + model_name)

                    for per in mvcc_dev_info.SpecialInfo.stCMLInfo.chSerialNumber:
                        if per == 0:
                            break
                        serial_number = serial_number + chr(per)
                    print("user serial number: " + serial_number)
                elif mvcc_dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
                    print("\nCXP device: [%d]" % i)
                    user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stCXPInfo.chUserDefinedName)
                    model_name = decoding_char(mvcc_dev_info.SpecialInfo.stCXPInfo.chModelName)
                    print("device user define name: " + user_defined_name)
                    print("device model name: " + model_name)

                    for per in mvcc_dev_info.SpecialInfo.stCXPInfo.chSerialNumber:
                        if per == 0:
                            break
                        serial_number = serial_number + chr(per)
                    print("user serial number: " + serial_number)
                elif mvcc_dev_info.nTLayerType == MV_GENTL_XOF_DEVICE:
                    print("\nXoF device: [%d]" % i)
                    user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stXoFInfo.chUserDefinedName)
                    model_name = decoding_char(mvcc_dev_info.SpecialInfo.stXoFInfo.chModelName)
                    print("device user define name: " + user_defined_name)
                    print("device model name: " + model_name)

                    for per in mvcc_dev_info.SpecialInfo.stXoFInfo.chSerialNumber:
                        if per == 0:
                            break
                        serial_number = serial_number + chr(per)
                    print("user serial number: " + serial_number)

                button_by_id = cam_button_group.button(i)
                button_by_id.setText("(" + serial_number + ")" + model_name)
                button_by_id.setEnabled(True)
                valid_number = valid_number + 1
            else:
                button_by_id = cam_button_group.button(i)
                button_by_id.setEnabled(False)

    def cam_check_box_clicked():
        global cam_checked_list
        cam_checked_list = []
        for i in range(0, 4):
            button = cam_button_group.button(i)
            if button.isChecked() is True:
                cam_checked_list.append(True)
            else:
                cam_checked_list.append(False)

    def enable_ui_controls():
        global b_is_open
        global b_is_grab
        global b_is_trigger
        global b_is_software_trigger
        ui.pushButton_enum.setEnabled(not b_is_open)
        ui.pushButton_open.setEnabled(not b_is_open)
        ui.pushButton_close.setEnabled(b_is_open)
        result1 = False if b_is_grab else b_is_open
        result2 = b_is_open if b_is_grab else False
        ui.pushButton_startGrab.setEnabled(result1)
        ui.pushButton_stopGrab.setEnabled(result2)
        ui.pushButton_saveImg.setEnabled(result2)
        ui.radioButton_continuous.setEnabled(b_is_open)
        ui.radioButton_trigger.setEnabled(b_is_open)
        ui.pushButton_setParams.setEnabled(b_is_open)
        ui.lineEdit_gain.setEnabled(b_is_open)
        ui.lineEdit_frameRate.setEnabled(b_is_open)
        ui.lineEdit_exposureTime.setEnabled(b_is_open)
        result3 = b_is_open if b_is_trigger else False
        ui.pushButton_triggerOnce.setEnabled(b_is_software_trigger and result3)
        ui.checkBox_software_trigger.setEnabled(b_is_trigger)

    def open_devices():
        global deviceList
        global obj_cam_operation
        global b_is_open
        global valid_number
        global cam_checked_list
        b_checked = 0
        if b_is_open is True:
            return

        if len(cam_checked_list) <= 0:
            print_text("please select a camera !")
            return
        obj_cam_operation = []
        for i in range(0, 4):
            if cam_checked_list[i] is True:
                b_checked = True
                camObj = MvCamera()
                obj_cam_operation.append(CameraOperation(camObj, deviceList, i))
                ret = obj_cam_operation[i].open_device()
                if 0 != ret:
                    obj_cam_operation.pop()
                    print_text("open cam %d fail ret[0x%x]" % (i, ret))
                    continue
                else:
                    b_is_open = True
            else:
                obj_cam_operation.append(0)
        if b_checked is False:
            print_text("please select a camera !")
            return
        if b_is_open is False:
            print_text("no camera opened successfully !")
            return
        else:
            ui.radioButton_continuous.setChecked(True)
            enable_ui_controls()

        for i in range(0, 4):
            if(i < valid_number) is True:
                button_by_id = cam_button_group.button(i)
                button_by_id.setEnabled(not b_is_open)

    def software_trigger_check_box_clicked():
        global obj_cam_operation
        global b_is_software_trigger
        if (ui.checkBox_software_trigger.isChecked()) is True:
            b_is_software_trigger = True
            for i in range(0, 4):
                if obj_cam_operation[i] != 0:
                    ret = obj_cam_operation[i].set_trigger_source("software")
                    if 0 != ret:
                        print_text('camera' + str(i) + ' set trigger source: software  fail! ret = ' + ToHexStr(ret))
        else:
            b_is_software_trigger = False
            for i in range(0, 4):
                if obj_cam_operation[i] != 0:
                    ret = obj_cam_operation[i].set_trigger_source("hardware")
                    if 0 != ret:
                        print_text('camera' + str(i) + ' set trigger source: hardware  fail! ret = ' + ToHexStr(ret))
        enable_ui_controls()

    def radio_button_clicked(button):
        global obj_cam_operation
        global b_is_trigger
        button_id = raio_button_group.id(button)
        if (button_id == 0) is True:
            b_is_trigger = False
            for i in range(0, 4):
                if obj_cam_operation[i] != 0:
                    ret = obj_cam_operation[i].set_trigger_mode("continuous")
                    if 0 != ret:
                        print_text('camera' + str(i) + ' set trigger mode: continuous fail! ret = ' + ToHexStr(ret))
            enable_ui_controls()

        else:
            b_is_trigger = True
            for i in range(0, 4):
                if obj_cam_operation[i] != 0:
                    ret = obj_cam_operation[i].set_trigger_mode("triggermode")
                    if 0 != ret:
                        print_text('camera' + str(i) + ' set trigger on fail! ret = ' + ToHexStr(ret))
            enable_ui_controls()

    def close_devices():
        global b_is_open
        global obj_cam_operation
        global valid_number

        if b_is_open is False:
            return
        if b_is_grab is True:
            stop_grabbing()
        for i in range(0, 4):
            if obj_cam_operation[i] != 0:
                ret = obj_cam_operation[i].close_device()
                if 0 != ret:
                    print_text('camera' + str(i) + ' close device fail! ret = ' + ToHexStr(ret))

            if i < valid_number:
                button_by_id = cam_button_group.button(i)
                button_by_id.setEnabled(True)
        b_is_open = False
        enable_ui_controls()

    def start_grabbing():
        global obj_cam_operation
        global win_display_handles
        global b_is_open
        global b_is_grab

        if (not b_is_open) or (b_is_grab is True):
            return

        for i in range(0, 4):
            if obj_cam_operation[i] != 0:
                ret = obj_cam_operation[i].start_grabbing(i, win_display_handles[i])
                if 0 != ret:
                    print_text('camera' + str(i) + ' start grabbing fail! ret = ' + ToHexStr(ret))
                b_is_grab = True
        enable_ui_controls()

    def stop_grabbing():
        global b_is_grab
        global obj_cam_operation
        global b_is_open

        if (not b_is_open) or (b_is_grab is False):
            return
        for i in range(0, 4):
            if obj_cam_operation[i] != 0:
                ret = obj_cam_operation[i].stop_grabbing()
                if 0 != ret:
                    print_text('camera' + str(i) + ' stop grabbing fail!ret = ' + ToHexStr(ret))
                b_is_grab = False
        enable_ui_controls()

    # ch:存图 | en:save image
    def save_bmp():
        global b_is_grab
        global obj_cam_operation

        if b_is_grab is False:
            return
        for i in range(0, 4):
            if obj_cam_operation[i] != 0:
                ret = obj_cam_operation[i].save_bmp()
                if 0 != ret:
                    print_text('camera' + str(i) + ' save bmp fail!ret = ' + ToHexStr(ret))

    def is_float(str_value):
        try:
            float(str_value)
            return True
        except ValueError:
            return False

    def set_parameters():
        global obj_cam_operation
        global b_is_open
        if b_is_open is False:
            return

        frame_rate = ui.lineEdit_frameRate.text()
        exposure_time = ui.lineEdit_exposureTime.text()
        gain = ui.lineEdit_gain.text()

        if is_float(frame_rate) is False or is_float(exposure_time) is False or is_float(gain) is False:
            print_text("parameters is valid, please check")
            return

        for i in range(0, 4):
            if obj_cam_operation[i] != 0:
                ret = obj_cam_operation[i].set_exposure_time(exposure_time)
                if ret != 0:
                    print_text('camera' + str(i) + ' Set exposure time failed ret:' + ToHexStr(ret))
                ret = obj_cam_operation[i].set_gain(gain)
                if ret != 0:
                    print_text('camera' + str(i) + ' Set gain failed ret:' + ToHexStr(ret))
                ret = obj_cam_operation[i].set_frame_rate(frame_rate)
                if ret != 0:
                    print_text('camera' + str(i) + ' set acquisition frame rate failed ret:' + ToHexStr(ret))

    def software_trigger_once():
        for i in range(0, 4):
            if obj_cam_operation[i] != 0:
                ret = obj_cam_operation[i].trigger_once()
                if ret != 0:
                    print_text('camera' + str(i) + 'TriggerSoftware failed ret:' + ToHexStr(ret))

    # ch: 初始化app, 绑定控件与函数 | en: Init app, bind ui and api
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    ui.pushButton_enum.clicked.connect(enum_devices)
    ui.pushButton_open.clicked.connect(open_devices)
    ui.pushButton_close.clicked.connect(close_devices)
    ui.pushButton_startGrab.clicked.connect(start_grabbing)
    ui.pushButton_stopGrab.clicked.connect(stop_grabbing)
    ui.pushButton_saveImg.clicked.connect(save_bmp)
    ui.pushButton_setParams.clicked.connect(set_parameters)
    ui.checkBox_software_trigger.clicked.connect(software_trigger_check_box_clicked)
    ui.pushButton_triggerOnce.clicked.connect(software_trigger_once)
    cam_button_group = QButtonGroup(mainWindow)
    cam_button_group.addButton(ui.checkBox_1, 0)
    cam_button_group.addButton(ui.checkBox_2, 1)
    cam_button_group.addButton(ui.checkBox_3, 2)
    cam_button_group.addButton(ui.checkBox_4, 3)

    cam_button_group.setExclusive(False)
    cam_button_group.buttonClicked.connect(cam_check_box_clicked)

    raio_button_group = QButtonGroup(mainWindow)
    raio_button_group.addButton(ui.radioButton_continuous, 0)
    raio_button_group.addButton(ui.radioButton_trigger, 1)
    raio_button_group.buttonClicked.connect(radio_button_clicked)

    win_display_handles.append(ui.widget_display1.winId())
    win_display_handles.append(ui.widget_display2.winId())
    win_display_handles.append(ui.widget_display3.winId())
    win_display_handles.append(ui.widget_display4.winId())

    mainWindow.show()
    enum_devices()
    enable_ui_controls()

    app.exec_()

    close_devices()

    # ch:反初始化SDK | en: finalize SDK
    MvCamera.MV_CC_Finalize()

    sys.exit()
