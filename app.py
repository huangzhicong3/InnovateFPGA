'''AI 监视器程序 主窗口'''

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


import cv2
import numpy as np
import logging as log
import time
from openvino.inference_engine import IENetwork, IEPlugin

import os
import os.path as osp
import sys

from widgets.loop_mode_input_dialog import LoopModeInputDialog
from widgets.display_widget import DisplayWidget
from constants import *
from debug.debug import debug

import pandas as pd
__appname__ = 'AI Monitor'
__appversion__ = '0.1.0'




class MainWindow(QMainWindow):

    def __init__(self):
        '''初始化函数，在整个程序生命周期中只会被调用一次'''
        super().__init__()

        # 全局数据池内容
        self.id_list = []
        self.character_list = []
        # 循环模式
        self.loop_mode = None

        self.initSubWidgets()
        self.initUI()
        self.display_widget.initPara()

    def initSubWidgets(self):
        '''初始化子窗口，绑定信号'''
        self.display_widget = DisplayWidget(self)

    def initUI(self):
        '''初始化UI，是__init__()的一部分'''

        # 设置尺寸策略
        widget_size_policy = QSizePolicy()
        widget_size_policy.setHorizontalPolicy(QSizePolicy.Minimum)
        widget_size_policy.setVerticalPolicy(QSizePolicy.Minimum)

        # 布局主窗口
        self.central = QWidget()
        self.layout = QGridLayout()

        # 初始化控件
        self.status_label = QLabel('待机中')
        self.start_or_reset_button = QPushButton('开始')
        self.close_button = QPushButton('退出')
        self.set_loop_mode_button =  QPushButton('设置循环模式')
        self.face_mode_button = QPushButton('人脸监控')
        self.passerby_mode_button = QPushButton('行人监控')
        self.fire_mode_button = QPushButton('火情监控')

        self.id_input_line = QLineEdit()
        self.id_input_line.setPlaceholderText('输入要登记的id')

        self.alarm_input_line = QLineEdit()
        self.alarm_input_line.setPlaceholderText('输入行人预警区域（1-6）')

        self.register_button = QPushButton('登记')

        self.pass_alarm_button = QPushButton('行人预警')

        self.logo_label = QLabel()
        self.logo_label.setPixmap(QPixmap(osp.abspath(MAIN_ICON)))
        self.logo_label.setScaledContents(True)

        # 绑定函数
        self.start_or_reset_button.clicked.connect(self.start_or_reset)
        self.close_button.clicked.connect(self.quit)
        self.set_loop_mode_button.clicked.connect(self.set_loop_mode)
        self.face_mode_button.clicked.connect(self.set_display_mode)
        self.passerby_mode_button.clicked.connect(self.set_display_mode)
        self.fire_mode_button.clicked.connect(self.set_display_mode)

        self.register_button.clicked.connect(self.register)

        self.pass_alarm_button.clicked.connect(self.set_alarm)

        # 放置控件
        self.layout.addWidget(self.display_widget, 0, 0, 22, 15)
        self.display_widget.setSizePolicy(widget_size_policy)

        self.layout.addWidget(self.status_label, 20, 1, 1, 1)
        self.status_label.setSizePolicy(widget_size_policy)

        self.status_label.setStyleSheet('color: white')

        self.layout.addWidget(self.start_or_reset_button, 3, 15, 2, 3)
        self.start_or_reset_button.setSizePolicy(widget_size_policy)

        self.layout.addWidget(self.close_button, 20, 15, 2, 3)
        self.close_button.setSizePolicy(widget_size_policy)

        self.layout.addWidget(self.set_loop_mode_button, 17, 15, 2, 3)
        self.set_loop_mode_button.setSizePolicy(widget_size_policy)

        self.layout.addWidget(self.face_mode_button, 6, 15, 1, 3)
        self.face_mode_button.setSizePolicy(widget_size_policy)
        self.layout.addWidget(self.id_input_line, 7, 15, 1, 3)
        self.id_input_line.setSizePolicy(widget_size_policy)
        self.layout.addWidget(self.register_button, 8, 15, 1, 3)
        self.register_button.setSizePolicy(widget_size_policy)

        self.layout.addWidget(self.passerby_mode_button, 10, 15, 1, 3)
        self.passerby_mode_button.setSizePolicy(widget_size_policy)
        self.layout.addWidget(self.alarm_input_line, 11, 15, 1, 3)
        self.alarm_input_line.setSizePolicy(widget_size_policy)
        self.layout.addWidget(self.pass_alarm_button, 12, 15, 1, 3)
        self.pass_alarm_button.setSizePolicy(widget_size_policy)

        self.layout.addWidget(self.fire_mode_button, 14, 15, 2, 3)
        self.fire_mode_button.setSizePolicy(widget_size_policy)





        self.layout.addWidget(self.logo_label, 0, 15, 3, 3)
        self.logo_label.setSizePolicy(widget_size_policy)

        self.central.setLayout(self.layout)
        self.setCentralWidget(self.central)
    def quit(self):
        self.source_flag = 0
        self.init_flag = 1
        self.close()
        sys.exit(0)
    def start_or_reset(self):
        self.loop_mode = None
        self.display_widget.init_source(sig=self.sender().text())
        self.start_or_reset_button.setText('复位')
        self.status_label.setText('AI未启动')

    def set_loop_mode(self):
        input_dialog = LoopModeInputDialog(self)
        input_dialog.exec_()
        if self.loop_mode:
            self.status_label.setText('循环模式')

    def set_display_mode(self):
        print('triggered')
        self.loop_mode = None
        if '人脸' in self.sender().text():
            if self.display_widget.source_flag != 1:
                self.status_label.setText('人脸模式')
                self.display_widget.init_flag = 0
                self.display_widget.source_flag = 1

        if '行人' in self.sender().text():
            if self.display_widget.source_flag != 2:
                self.status_label.setText('行人模式')
                self.display_widget.init_flag = 0
                self.display_widget.source_flag = 2

        if '火情' in self.sender().text():
            if self.display_widget.source_flag != 3:
                self.status_label.setText('火情模式')
                self.display_widget.init_flag = 0
                self.display_widget.source_flag = 3

    def register(self):
        '''TODO 函数中的the_id变量已经获取id文本，请实现id输入后的程序行为'''
        the_id = self.id_input_line.text()
        self.display_widget.regist(the_id)
        pass
    def set_alarm(self):
        alarm_area = int(self.alarm_input_line.text())
        self.display_widget.alarm_dete(alarm_area)
        pass