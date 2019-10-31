'''main.py是最终程序打包的起点'''

import sys

from debug.debug import debug

from PyQt5.QtWidgets import *
from PyQt5.Qt import QIcon

from app import __appname__, __appversion__, MainWindow
from constants import MAIN_ICON

def main():
    app = QApplication(sys.argv)

    app.setApplicationName(__appname__)
    app.setApplicationVersion(__appversion__)

    win = MainWindow()
    win.show()
    win.raise_()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
