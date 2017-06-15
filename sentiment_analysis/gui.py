#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QLineEdit, QLabel
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Анализатор'
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.button = QPushButton('Выкачать твиты', self)
        self.button.setToolTip('This is an example button')
        self.button.move(100, 70)
        self.button.clicked.connect(self.on_click)

        self.edit = QLineEdit(self)
        self.edit.setText('Введите #хэштег')
        self.edit.move(100,30)

        self.label = QListWidget(self)
        self.label.resize(100,40)
        self.label.move(100, 120)

        self.label1 = QListWidget(self)
        self.label1.resize(100, 40)
        self.label1.move(400, 120)

        self.show()

    @pyqtSlot()
    def on_click(self):

        print('PyQt5 button click')
        f = open('predobrabot.txt', 'r', encoding='utf8')
        k = 0
        data = []
        for line in f.readlines():
            data.append(line)
        data = np.array(data)

        x=[]
        x.append(data[1])
        x.append(data[2])
        x=np.array(x)
        print(x)
        f.close()
        self.label.addItem(QListWidgetItem(x[1]))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())