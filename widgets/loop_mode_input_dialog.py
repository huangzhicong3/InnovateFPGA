
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from constants import *
from debug.debug import debug

import os.path as osp
import time

class LoopModeInputDialog(QDialog):

    confirm_signal = pyqtSignal(object)

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window
        self.initUI()

    def initUI(self):
        with debug('初始化输入窗口'):
            self.setWindowTitle('循环设定')
            self.resize(150,100)

            self.layout = QGridLayout()

            # 初始化和命名控件
            self.label_1 = QLabel('人脸监控时间（秒）')
            self.label_2 = QLabel('行人监控时间（秒）')
            self.label_3 = QLabel('火情监控时间（秒）')
            self.line_1 = QLineEdit()
            self.line_2 = QLineEdit()
            self.line_3 = QLineEdit()
            self.line_1.setObjectName('face')
            self.line_2.setObjectName('passerby')
            self.line_3.setObjectName('fire')
            self.line_1.setValidator(QIntValidator())
            self.line_2.setValidator(QIntValidator())
            self.line_3.setValidator(QIntValidator())

            self.confirm_button = QPushButton('确定')
            self.confirm_button.clicked.connect(self.confirm)
            self.cancel_button = QPushButton('取消')
            self.cancel_button.clicked.connect(self.cancel)

            # 放置控件
            self.label_1.setSizePolicy(MAX_SIZE_POLICY)
            self.label_2.setSizePolicy(MAX_SIZE_POLICY)
            self.label_3.setSizePolicy(MAX_SIZE_POLICY)
            self.line_1.setSizePolicy(MAX_SIZE_POLICY)
            self.line_2.setSizePolicy(MAX_SIZE_POLICY)
            self.line_3.setSizePolicy(MAX_SIZE_POLICY)
            self.confirm_button.setSizePolicy(MAX_SIZE_POLICY)
            self.cancel_button.setSizePolicy(MAX_SIZE_POLICY)

            self.layout.addWidget(self.label_1, 0, 0, 1, 1)
            self.layout.addWidget(self.label_2, 1, 0, 1, 1)
            self.layout.addWidget(self.label_3, 2, 0, 1, 1)
            self.layout.addWidget(self.line_1, 0, 1, 1, 1)
            self.layout.addWidget(self.line_2, 1, 1, 1, 1)
            self.layout.addWidget(self.line_3, 2, 1, 1, 1)
            self.layout.addWidget(self.confirm_button, 3, 0, 1, 1)
            self.layout.addWidget(self.cancel_button, 3, 1, 1, 1)

            self.setLayout(self.layout)

    def confirm(self):
        '''确定键功能，进行输入信息的合法性检查，不合法则提示用户重新输入，合法则将输入信息发送回主窗口，然后关闭输入窗口'''
        valid, report = self.set_loop_mode()
        if not valid:
            QMessageBox.warning(self, '输入有误', report)
        else:
            self.close()

    def set_loop_mode(self):
        with debug('set loop'):

            report = ''
            face_time = self.findChild(QLineEdit, 'face').text()
            passerby_time = self.findChild(QLineEdit, 'passerby').text()
            fire_time = self.findChild(QLineEdit, 'fire').text()
            valid1 =  (face_time != '') and (passerby_time != '') and (fire_time != '')
            valid2 = (int(face_time) > 10) and (int(passerby_time) > 10) and (int(fire_time) > 10)
            valid = valid1 and valid2

            if valid:
                self.main_window.loop_mode = [int(face_time), int(passerby_time), int(fire_time), time.time()]
                return valid, ''
            else:
                if not valid1:
                    report += '所有模式的的时间都要填写，不启用填写0\n'
                if not valid2:
                    report += '所有模式的的时间都要大于10\n'

                return valid, report

    def cancel(self):
        self.close()

if __name__ == '__main__':
    pass
