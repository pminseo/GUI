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
        self.histogramAct = QAction("Hostogram Equalization")
        self.add2imgsAct = QAction("Addition of two images")
        self.sub2imgsAct = QAction("Subtraction of two images")
        
        self.imgArea = QWidget()
        self.inImg = 0
        self.inImg2 = 0
        self.outImg = 0
        
        self.inImgLabel = QLabel()
        self.inImgLabel2 = QLabel()
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
        self.imgMenu.addAction(self.histogramAct)
        self.imgMenu.addAction(self.add2imgsAct)
        self.imgMenu.addAction(self.sub2imgsAct)
        
        self.mBar.addMenu(self.fileMenu)
        self.mBar.addMenu(self.imgMenu)
        
        self.openAct.triggered.connect(self.openFile)
        self.saveAct.triggered.connect(self.saveFile)
        self.exitAct.triggered.connect(self.quitApp)
        
        self.binaryAct.triggered.connect(self.binarization)
        self.arithAct.triggered.connect(self.arithmetic)
        self.edgeAct.triggered.connect(self.edgeDetection)
        self.histogramAct.triggered.connect(self.histogramEqual)
        self.add2imgsAct.triggered.connect(self.add2Images)
        self.sub2imgsAct.triggered.connect(self.sub2Images)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.inImgLabel)
        hbox.addWidget(self.inImgLabel2)
        hbox.addWidget(self.outImgLabel)
        self.imgArea.setLayout(hbox)
        
        self.setCentralWidget(self.imgArea)
        self.setMenuBar(self.mBar)
        self.setStatusBar(self.sBar)
        
        self.setFixedSize(540,300)
        self.viewMode = 2
        
        self.show()

    def openFile(self):
        fname = self.loadFile()

        # print(fname)
        # print(fname[0])
        # # im = Image.open(fname[0])
        # # if fname[0].split(".")[-1] in ['pgm', 'ppm', 'PGM', 'PPM']:
        # #     im = im.resize((256,256))
        #     # print(type(im))
        # 
        # # self.inImg = np.array(im)
        # self.inImg = self.fileopen(fname[0])

        im = Image.open(fname[0])
        if fname[0].split(".")[-1] in ['pgm', 'ppm', 'PGM', 'PPM']: im = im.resize((256,256))
        
        self.inImg = np.array(im)
        # print(self.inImg)
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
            
        if self.viewMode != 2:
            self.viewMode = 2
            self.setviewMode(w,h)
        
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
        
        if self.viewMode != 2:
            self.viewMode = 2
            self.setviewMode(w,h)
            
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
        
        if self.viewMode != 2:
            self.viewMode = 2
            self.setviewMode(w,h)
            
        qImg = QImage(self.inImg.data, w,h, QImage.Format_Grayscale8)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
        d = CheckWidget(self, self.pos())
        d.command.connect(self.rbClicked)
        
    def rbClicked(self, buttonId):  # X = Vertical, Y = Horizontal
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
        
    def histogramEqual(self):
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
            
        if self.viewMode != 2:
            self.viewMode = 2
            self.setviewMode(w,h)
        self.outImg = self.inImg.copy()
        hist = [0 for i in range(256)]
        acc_hist = 0
        for j in range(h):
            for i in range(w):
                k = self.inImg[i][j]
                hist[k] += 1
        cumulativeSum = []
        for i in range(256):
            acc_hist += hist[i]
            cumulativeSum.append(acc_hist)
        for j in range(h):
            for i in range(w):
                k = self.inImg[i][j]
                self.outImg[i][j] = cumulativeSum[k]/(256*256)*255
                
        qImg = QImage(self.outImg.data, w,h, QImage.Format_Grayscale8)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
    def setviewMode(self, w, h):
        self.setFixedSize((w + 14) * self.viewMode, h + 44)
        if self.viewMode == 2:
            self.inImg2 = 0
            self.inImgLabel2.setPixmap(QPixmap())
    
    def loadFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Select the First Image', './', "BMP files (*.bmp);; PGM files (*.pgm);; JPG files(*.jpg);; All files(*.*)")
        return fname
    
    def fileopen(self, file_name):
        with open(file_name, 'rb') as f:
            file_type = file_name.split('.')[-1]
            if file_type == 'pgm' or file_type == 'PGM':
                pgm_type = f.readline()
                
                wh_line = f.readline().decode('utf-8').split()
                while wh_line[0] == '#':
                    wh_line = f.readline().split()
                (img_width, img_height) = [int(i) for i in wh_line]
                print('width: {}, height: {}'.format(img_width, img_height))
                max_value = f.readline()
                while max_value[0] == '#':
                    max_value = f.readline()
                max_value = int(max_value)
                print('PGM type: {}'.format(pgm_type.decode('utf-8')))
                if pgm_type == b'P5\n':
                    img_depth = 1
                else:
                    img_depth = 3
            
            elif file_type == 'raw' or file_type == 'RAW':
                raw_data = f.readline()
                assert len(bytearray(raw_data)) != 256*256, "Only 256 * 256 image"
                img_width, img_height, img_depth = 256, 256, 1
            col = []
            for i in range(img_height):
                row = []
                for j in range(img_width):
                    row.append(list(f.readline(1 * img_depth)))
                col.append(row)
            
            result = np.array(col)
            print(result.shape)
            # result = np.array(result).squeeze(-1).transpose(1,2,0)
        return result
    
    def add2Images(self):
        self.viewMode = 3
        fname = self.loadFile()
        im = Image.open(fname[0])
        if fname[0].split(".")[-1] in ['pgm', 'ppm', 'PGM', 'PPM']: im = im.resize((256,256))

        self.inImg = np.array(im)
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        self.setviewMode(w,h)
        qImg = QImage(self.inImg.data, w, h, QImage.Format_Grayscale8)
        self.inImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
        fname2 = self.loadFile()
        im2 = Image.open(fname2[0])
        if w != im2.size[0] or h != im2.size[1]:
            im2 = im2.resize((w,h))
        self.inImg2 = np.array(im2)
        
        qImg2 = QImage(self.inImg2.data, w, h, QImage.Format_Grayscale8)
        self.inImgLabel2.setPixmap(QPixmap.fromImage(qImg2))
        self.outImg = self.inImg.copy()
        
        alpha, beta = 1, 1
        for j in range(h):
            for i in range(w):
                val = int((alpha * self.inImg[j][i]) + (beta * self.inImg2[j][i]))
                self.outImg[j][i] = 255 if val > 255 else val
                
        qImg3 = QImage(self.outImg.data, w, h, QImage.Format_Grayscale8)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg3))
        
    def sub2Images(self):
        self.viewMode = 3
        fname = self.loadFile()
        im = Image.open(fname[0])
        if fname[0].split(".")[-1] in ['pgm', 'ppm', 'PGM', 'PPM']: im = im.resize((256,256))

        self.inImg = np.array(im)
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        self.setviewMode(w,h)
        qImg = QImage(self.inImg.data, w, h, QImage.Format_Grayscale8)
        self.inImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
        fname2 = self.loadFile()
        im2 = Image.open(fname2[0])
        if w != im2.size[0] or h != im2.size[1]:
            im2 = im2.resize((w,h))
        self.inImg2 = np.array(im2)
        
        qImg2 = QImage(self.inImg2.data, w, h, QImage.Format_Grayscale8)
        self.inImgLabel2.setPixmap(QPixmap.fromImage(qImg2))
        self.outImg = self.inImg.copy()
        
        print(self.inImg.shape, self.inImg2.shape)

        alpha, beta = 1, 1
        for j in range(h):
            for i in range(w):
                val = int((alpha * self.inImg[j][i]) - (beta * self.inImg2[j][i]))
                self.outImg[j][i] = 0 if val < 0 else val
                
        qImg3 = QImage(self.outImg.data, w, h, QImage.Format_Grayscale8)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg3))
        
if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        ex = ImageProcessing()
        sys.exit(app.exec_())
    except:
        pass
    