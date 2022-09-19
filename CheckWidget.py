import sys

from PyQt5.QtWidgets import QVBoxLayout, QApplication, QWidget, QButtonGroup, QRadioButton
from PyQt5.QtCore import Qt, pyqtSignal

from PIL import Image
import numpy as np

class CheckWidget(QWidget):
    command = pyqtSignal(int)
    
    def __init__(self, parent, mainWindowPos):
        super(CheckWidget, self).__init__(parent, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        self.mainWindowPos = mainWindowPos
        self.bgWidget = QWidget(self)
        self.bg = QButtonGroup(self.bgWidget)
        
        self.rb1 = QRadioButton("Roberts", self.bgWidget)
        self.rb2 = QRadioButton("Prewitt", self.bgWidget)
        self.rb3 = QRadioButton("Sobel", self.bgWidget)
        
        self.initUI()
        
    def initUI(self):
        self.rb1.move(10, 10)
        self.rb2.move(10, 40)
        self.rb3.move(10, 70)
        self.bg.addButton(self.rb1, 0)
        self.bg.addButton(self.rb2, 1)
        self.bg.addButton(self.rb3, 2)
        
        self.bg.buttonClicked.connect(self.rbChecked)
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.rb1)
        vbox.addWidget(self.rb2)
        vbox.addWidget(self.rb3)
        self.setLayout(vbox)
        
        self.setWindowTitle("Edge Detection")
        self.move(self.mainWindowPos.x() + 540, self.mainWindowPos.y())
        self.show()
        
    def rbChecked(self):
        self.command.emit(self.bg.checkedId())
    