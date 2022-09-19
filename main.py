import sys

from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFileDialog
from PyQt5.QtWidgets import QMainWindow, QMenuBar, QMenu, QAction, QStatusBar
from PyQt5.QtGui import QPixmap, QImage

from PIL import Image
import numpy as np
import math

from SliderWidget import SliderWidget
from CheckWidget import CheckWidget

class ImageProcessing(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mBar = QMenuBar()
        self.sBar = QStatusBar()
        
        self.fileMenu = QMenu("File")
        self.openAct = QAction("Open")
        self.saveAct = QAction("Save")
        self.exitAct = QAction("Exit")
        
        self.imgMenu = QMenu("Image Processing")
        self.binaryAct = QAction("Binarization")
        self.arithAct = QAction("Arithmetic Operation")
        self.edgeAct = QAction("Edge Detection")
        
        self.imgArea = QWidget()
        self.inImg = 0
        self.outImg = 0
        
        self.inImgLabel = QLabel()
        self.outImgLabel = QLabel()
        
        self.initUI()
        
    def initUI(self):
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.saveAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)
        
        self.imgMenu.addAction(self.binaryAct)
        self.imgMenu.addAction(self.arithAct)
        self.imgMenu.addAction(self.edgeAct)
        
        self.mBar.addMenu(self.fileMenu)
        self.mBar.addMenu(self.imgMenu)
        
        self.openAct.triggered.connect(self.openFile)
        self.saveAct.triggered.connect(self.saveFile)
        self.exitAct.triggered.connect(self.quitApp)
        
        self.binaryAct.triggered.connect(self.binarization)
        self.arithAct.triggered.connect(self.arithmetic)
        self.edgeAct.triggered.connect(self.edgeDetection)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.inImgLabel)
        hbox.addWidget(self.outImgLabel)
        self.imgArea.setLayout(hbox)
        
        self.setCentralWidget(self.imgArea)
        self.setMenuBar(self.mBar)
        self.setStatusBar(self.sBar)
        
        self.setFixedSize(540,300)
        
        self.show()

    def openFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './', "BMP files (*.bmp);; JPG files(*.jpg);; All files(*.*)")
        im = Image.open(fname[0])
        
        self.inImg = np.array(im)
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        
        qImg = QImage(self.inImg.data, w, h, QImage.Format_Grayscale8)
        self.inImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
        self.sBar.showMessage('File Open : "' + fname[0] + '"')
        
    def saveFile(self):
        fname = QFileDialog.getSaveFileName(self, 'Save file', './', "BMP files (*.bmp);; JPG files(*.jpg);; All files(*.*)")
        
        w = self.outImg.shape[1]
        h = self.outImg.shape[0]
        
        qImg = QImage(self.outImg.data, w, h, QImage.Format_Grayscale8)
        
        if qImg.save(fname[0]):
            self.sBar.showMessage("Image Saved")
        else:
            self.sBar.showMessage("Image Save Error")
            
    def quitApp(self):
        self.close()
        
    def binarization(self):
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        
        self.outImg = self.inImg.copy()
        
        for y in range(h):
            for x in range(w):
                if self.outImg[y][x] > 128:
                    self.outImg[y][x] = 255
                else:
                    self.outImg[y][x] = 0
    
        qImg = QImage(self.outImg.data, w,h, QImage.Format_Grayscale8)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
    def arithmetic(self):
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        qImg = QImage(self.inImg.data, w,h, QImage.Format_Grayscale8)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
        s = SliderWidget(self, self.pos(), self.inImg)
        s.command.connect(self.sliderValueChanged)
        
    def sliderValueChanged(self, out):
        self.outImg = out
        w = self.outImg.shape[1]
        h = self.outImg.shape[0]
        
        qImg = QImage(self.outImg.data, w,h, QImage.Format_Grayscale8)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
    def edgeDetection(self):
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        qImg = QImage(self.inImg.data, w,h, QImage.Format_Grayscale8)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
        d = CheckWidget(self, self.pos())
        d.command.connect(self.rbClicked)
        
    def rbClicked(self, buttonId):
        robertsX = [[0,0,-1],[0,1,0],[0,0,0]]
        robertsY = [[-1,0,0],[0,1,0],[0,0,0]]
        
        prewittX = [[1,1,1],[0,0,0],[-1,-1,-1]]
        prewittY = [[-1,0,1],[-1,0,1],[-1,0,1]]
        
        sobelX = [[1,2,1],[0,0,0],[-1,-2,-1]]
        sobelY = [[-1,0,1],[-2,0,2],[-1,0,1]]
        
        if buttonId == 0:
            maskX = robertsX
            maskY = robertsY
        elif buttonId == 1:
            maskX = prewittX
            maskY = prewittY
        elif buttonId == 2:
            maskX = sobelX
            maskY = sobelY
        else:
            maskX = [[0,0,0],[0,0,0],[0,0,0]]
            maskY = [[0,0,0],[0,0,0],[0,0,0]]
        
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        
        self.outImg = np.zeros((h,w), dtype=np.uint8)
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                tempX = np.multiply(maskX, self.inImg[y-1:y+2, x-1:x+2])
                tempY = np.multiply(maskY, self.inImg[y-1:y+2, x-1:x+2])
                
                temp = int(math.sqrt(np.sum(tempX)**2 + np.sum(tempY)**2))
                
                if temp > 255:
                    temp = 255
                if temp < 0:
                    temp = 0
                
                self.outImg[y][x] = temp
                
        qImg = QImage(self.outImg.data, w, h, QImage.Format_Grayscale8)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        ex = ImageProcessing()
        sys.exit(app.exec_())
    except:
        pass
    