import sys

from PyQt5.QtWidgets import QVBoxLayout, QWidget, QSlider, QLabel
from PyQt5.QtCore import Qt, pyqtSignal

from PIL import Image
import numpy as np

class SliderWidget(QWidget):
    command = pyqtSignal(np.ndarray)
    
    def __init__(self, parent, mainWindowPos, img):
        super(SliderWidget, self).__init__(parent, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        self.mainWindowPos = mainWindowPos
        self.img = img
        
        self.slider = QSlider(Qt.Horizontal)
        self.lbVal = QLabel("0")
        
        self.initUI()
        
    def initUI(self):
        self.slider.setTickPosition(QSlider.TicksAbove)
        self.slider.setTickInterval(10)
        self.slider.setRange(-255,255)
        self.slider.setMinimumWidth(520)
        self.slider.setValue(0)
        
        self.lbVal.setAlignment(Qt.AlignHCenter)
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.slider)
        vbox.addWidget(self.lbVal)
        
        self.slider.valueChanged.connect(self.changedValue)
        
        self.setWindowTitle("Arithmetic Operation")
        self.setLayout(vbox)
        self.move(self.mainWindowPos.x(), self.mainWindowPos.y()+330)
        self.show()
        
    def changedValue(self):
        self.lbVal.setText(str(self.slider.value()))
        
        outImg = self.img.copy()
        
        w = outImg.shape[1]
        h = outImg.shape[0]
        
        for y in range(h):
            for x in range(w):
                temp = outImg[y][x]
                temp += self.slider.value()
                if temp > 255:
                    temp = 255
                elif temp < 0:
                    temp = 0
                    
                outImg[y][x] = temp
            
        self.command.emit(outImg)
        