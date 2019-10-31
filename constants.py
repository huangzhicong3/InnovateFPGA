from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


MAIN_ICON = 'external/logo.jpg'
WAIT_IMAGE = './external/wait.jpg'

MIN_SIZE_POLICY = QSizePolicy()
MIN_SIZE_POLICY.setHorizontalPolicy(QSizePolicy.Minimum)
MIN_SIZE_POLICY.setVerticalPolicy(QSizePolicy.Minimum)

MAX_SIZE_POLICY = QSizePolicy()
MAX_SIZE_POLICY.setHorizontalPolicy(QSizePolicy.Maximum)
MAX_SIZE_POLICY.setVerticalPolicy(QSizePolicy.Maximum)

IGNORE_SIZE_POLICY = QSizePolicy()
IGNORE_SIZE_POLICY.setHorizontalPolicy(QSizePolicy.Ignored)
IGNORE_SIZE_POLICY.setVerticalPolicy(QSizePolicy.Ignored)