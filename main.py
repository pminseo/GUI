import sys

from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFileDialog
from PyQt5.QtWidgets import QMainWindow, QMenuBar, QMenu, QAction, QStatusBar
from PyQt5.QtGui import QPixmap, QImage

from PIL import Image
import numpy as np
import math

from SliderWidget import SliderWidget
from CheckWidget import CheckWidget, SharpeningWidget, AveragingWidget

from Interpolation import nn_interpolate


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
        self.sharpAct = QAction("Sharpening")
        self.avgmaskAct = QAction("Averaging")
        self.medianAct = QAction("Median Filtering")
        self.erosionAct = QAction("Erosion")
        self.dilationAct = QAction("Dilation")
        self.openingAct = QAction("Opening")
        self.closingAct = QAction("Closing")


        self.visited = [False]
        
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
        self.imgMenu.addSeparator()
        self.imgMenu.addAction(self.histogramAct)
        self.imgMenu.addAction(self.add2imgsAct)
        self.imgMenu.addAction(self.sub2imgsAct)
        self.imgMenu.addSeparator()
        self.imgMenu.addAction(self.edgeAct)
        self.imgMenu.addAction(self.sharpAct)
        self.imgMenu.addAction(self.avgmaskAct)
        self.imgMenu.addAction(self.medianAct)
        self.imgMenu.addSeparator()
        self.imgMenu.addAction(self.erosionAct)
        self.imgMenu.addAction(self.dilationAct)
        self.imgMenu.addAction(self.openingAct)
        self.imgMenu.addAction(self.closingAct)

        
        
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
        self.sharpAct.triggered.connect(self.sharpening)
        self.avgmaskAct.triggered.connect(self.averaging)
        self.medianAct.triggered.connect(self.medianFiltering)
        self.erosionAct.triggered.connect(self.erosion)
        self.dilationAct.triggered.connect(self.dilation)
        self.openingAct.triggered.connect(self.opening)
        self.closingAct.triggered.connect(self.closing)

        
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
        im = self.pgm_processing(fname[0]) if fname[0].split('.')[-1] in ['pgm', 'ppm', 'raw', 'PGM', 'PPM', 'RAW'] else Image.open(fname[0])
        self.inImg = im if isinstance(im, np.ndarray) else np.array(im, dtype=np.uint8)

        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        
        self.setviewMode(w, h)
        qImg = QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_Grayscale8) if self.inImg.shape[-1] != 3 else QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qImg)
        self.inImgLabel.setPixmap(pixmap)
        self.outImgLabel.setPixmap(QPixmap())
        
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
        
    def binarization(self, *args, **kargs):
        viewOutput = kargs["viewOutput"] if "viewOutput" in kargs else True
        
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]

        if self.viewMode != 2: self.viewMode = 2
        self.setviewMode(w, h)

        self.outImg = self.inImg.copy()
        
        for i in range(h):
            for j in range(w):
                if self.outImg[i][j] > 128:
                    self.outImg[i][j] = 255
                else:
                    self.outImg[i][j] = 0
        
        if viewOutput:
            qImg = QImage(self.outImg.data, w,h, QImage.Format_Grayscale8)
            self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        else:
            self.inImg = self.outImg.copy()
            qImg = QImage(self.inImg.data, w,h, QImage.Format_Grayscale8)
            self.inImgLabel.setPixmap(QPixmap.fromImage(qImg))
            
        
    def arithmetic(self):
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        
        if self.viewMode != 2: self.viewMode = 2
        self.setviewMode(w,h)
            
        qImg = QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_Grayscale8) if self.inImg.shape[-1] != 3 else QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_RGB888)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
        s = SliderWidget(self, self.pos(), self.inImg)
        s.command.connect(self.sliderValueChanged)
        
    def sliderValueChanged(self, out):
        self.outImg = out
        w = self.outImg.shape[1]
        h = self.outImg.shape[0]
        
        qImg = QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_Grayscale8) if self.outImg.shape[-1] != 3 else QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_RGB888)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
    def edgeDetection(self):
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        
        if self.viewMode != 2: self.viewMode = 2
        self.setviewMode(w,h)
        
        qImg = QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_Grayscale8) if self.inImg.shape[-1] != 3 else QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_RGB888)
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
                
        qImg = QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_Grayscale8) if self.outImg.shape[-1] != 3 else QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_RGB888)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
    
    def sharpening(self):
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        
        if self.viewMode != 2: self.viewMode = 2
        self.setviewMode(w,h)
        
        qImg = QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_Grayscale8) if self.inImg.shape[-1] != 3 else QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_RGB888)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
        d = SharpeningWidget(self, self.pos())
        d.command.connect(self.sbClicked)
    
    def sbClicked(self, buttonId):
        mask1 = [[0,-1,0], [-1,5,-1], [0,-1,0]]
        mask2 = [[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]
        if buttonId == 0:
            mask = mask1
        elif buttonId == 1:
            mask = mask2
        else:
            mask = [[0,0,0],[0,0,0],[0,0,0]]
            
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        self.outImg = np.zeros((h,w), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                if i == 0 or i == h-1 or j == 0 or j == w-1:
                    self.outImg[i][j] = self.inImg[i][j]
                else:
                    temp = np.sum(np.multiply(mask, self.inImg[i-1:i+2, j-1:j+2]))
                    if temp > 255:
                        temp = 255
                    if temp < 0:
                        temp = 0
                    
                    self.outImg[i][j] = temp
        
        qImg = QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_Grayscale8) if self.outImg.shape[-1] != 3 else QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_RGB888)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
                    
    
    def averaging(self):
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        
        if self.viewMode != 2: self.viewMode = 2
        self.setviewMode(w,h)
        
        qImg = QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_Grayscale8) if self.inImg.shape[-1] != 3 else QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_RGB888)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
        d = AveragingWidget(self, self.pos())
        d.command.connect(self.abClicked)
    
    def abClicked(self, buttonId):
        if buttonId == 0:
            mask_size = 3
        elif buttonId == 1:
            mask_size = 5
        else:
            mask_size = 1
        
        t = int(mask_size//2)
        
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        
        self.outImg = np.zeros((h,w), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                if i < t or i >= h-t or j < t or j >= w-t:
                    self.outImg[i][j] = self.inImg[i][j]
                else:
                    temp = np.sum(self.inImg[i-t:i+t+1, j-t:j+t+1]) / (mask_size**2)
                    self.outImg[i][j] = temp
        
        qImg = QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_Grayscale8) if self.outImg.shape[-1] != 3 else QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_RGB888)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
    def histogramEqual(self):
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
            
        if self.viewMode != 2: self.viewMode = 2
        self.setviewMode(w,h)

        self.outImg = self.inImg.copy()
        hist = [0 for i in range(256)]
        acc_hist = 0
        for i in range(h):
            for j in range(w):
                k = self.inImg[i][j]
                hist[k] += 1
        cumulativeSum = []
        for i in range(256):
            acc_hist += hist[i]
            cumulativeSum.append(acc_hist)
        
        for i in range(h):
            for j in range(w):
                k = self.inImg[i][j]
                self.outImg[i][j] = cumulativeSum[k] / (256 * 256) * 255
                
        qImg = QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_Grayscale8) if self.outImg.shape[-1] != 3 else QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_RGB888)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
    
    def erosion(self, *args,**kargs):
        viewOutput = kargs["viewOutput"] if "viewOutput" in kargs else True
        
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]

        self.outImg = np.zeros((h,w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                if i == 0 or i == h-1 or j == 0 or j == w-1:
                    self.outImg[i][j] = self.inImg[i][j]
                else:
                    temp = self.inImg[i-1:i+2, j-1:j+2]
                    temp = np.sort(np.ravel(temp, order='C'))
                    self.outImg[i][j] = temp[0]
        
        if viewOutput:
            qImg = QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_Grayscale8) if self.outImg.shape[-1] != 3 else QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_RGB888)
            self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))


    def dilation(self, *args, **kargs):
        viewOutput = kargs["viewOutput"] if "viewOutput" in kargs else True

        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        
        self.outImg = np.zeros((h,w), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                if i == 0 or i == h-1 or j == 0 or j == w-1:
                    self.outImg[i][j] = self.inImg[i][j]
                else:
                    temp = self.inImg[i-1:i+2, j-1:j+2]
                    temp = np.sort(np.ravel(temp, order='C'))
                    self.outImg[i][j] = temp[-1]
        if viewOutput:
            qImg = QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_Grayscale8) if self.outImg.shape[-1] != 3 else QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_RGB888)
            self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))


    def CopyResult2Input(self, **kargs):
        viewInput = kargs["viewInput"] if "viewInput" in kargs else False
        self.inImg = self.outImg.copy()
        if viewInput:
            w, h = self.inImg.shape[1], self.inImg.shape[0]
            qImg = QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_Grayscale8) if self.inImg.shape[-1] != 3 else QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_RGB888)
            self.inImgLabel.setPixmap(QPixmap.fromImage(qImg))

    def opening(self):
        self.binarization(viewOutput=False)

        self.CopyResult2Input(viewInput=True)
        self.erosion(viewOutput=False)

        self.CopyResult2Input()
        self.erosion(viewOutput=False)

        self.CopyResult2Input()
        self.erosion(viewOutput=False)

        self.CopyResult2Input()
        self.dilation(viewOutput=False)

        self.CopyResult2Input()
        self.dilation(viewOutput=False)

        self.CopyResult2Input()
        self.dilation(viewOutput=True)

        self.countObjects()

    def closing(self):
        self.binarization(viewOutput=False)

        self.CopyResult2Input(viewInput=True)
        self.dilation(viewOutput=False)

        self.CopyResult2Input()
        self.dilation(viewOutput=False)

        self.CopyResult2Input()
        self.dilation(viewOutput=False)

        self.CopyResult2Input()
        self.erosion(viewOutput=False)

        self.CopyResult2Input()
        self.erosion(viewOutput=False)

        self.CopyResult2Input()
        self.erosion(viewOutput=True)

        self.countObjects()

    def countObjects(self):
        count = 0
        w, h = self.outImg.shape[1], self.outImg.shape[0]
        self.visited = [[False for _ in range(w)] for _ in range(h)]
        
        for i in range(h):
            for j in range(w):
                if self.outImg[i][j] == 255 and self.visited[i][j] == False:
                    count += 1
                    self.visited[i][j] = True
                    self.dfs(i-1, j)
                    self.dfs(i+1, j)
                    self.dfs(i, j-1)
                    self.dfs(i, j+1)
        print(count)
    
    def dfs(self, i, j):
        if i < 0 or i >= self.outImg.shape[0] or j < 0 or j >= self.outImg.shape[1] or self.visited[i][j] == True:
            return
        if self.outImg[i][j] == 255:
            self.visited[i][j] = True
            self.dfs(i-1, j)
            self.dfs(i+1, j)
            self.dfs(i, j-1)
            self.dfs(i, j+1)
        return


    def medianFiltering(self):
        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
            
        if self.viewMode != 2: self.viewMode = 2
        self.setviewMode(w,h)
        
        self.outImg = np.zeros((h,w), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                if i == 0 or i == h-1 or j == 0 or j == w-1:
                    self.outImg[i][j] = self.inImg[i][j]
                else:
                    temp = self.inImg[i-1:i+2, j-1:j+2]
                    temp = np.sort(np.ravel(temp, order='C'))
                    self.outImg[i][j] = temp[4]
        
        qImg = QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_Grayscale8) if self.outImg.shape[-1] != 3 else QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_RGB888)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg))
    
    
    def setviewMode(self, w, h):
        cw, cy = int(1.1 * w) * self.viewMode, int(1.2 * h)
        self.setFixedSize(cw, cy)
        if self.viewMode == 2:
            self.inImg2 = 0
            self.inImgLabel2.setPixmap(QPixmap())
    
    def loadFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Select the First Image', './', "BMP files (*.bmp);; PGM files (*.pgm);; JPG files(*.jpg);; All files(*.*)")
        return fname
    
    def pgm_processing(self, file_name):
        b_type = True
        file_type = file_name.split('.')[-1]
        with open(file_name, 'rb') as f:
            if file_type == 'pgm' or file_type == 'PGM':
                b_type = f.readline() == b"P5\n"
                wh_line = f.readline().decode('utf-8').split()
                while wh_line[0] == '#':
                    wh_line = f.readline().split()
                (img_width, img_height) = [int(i) for i in wh_line]
                # print('width: {}, height: {}'.format(img_width, img_height))
                max_value = f.readline()
                while max_value[0] == '#':
                    max_value = f.readline()
                max_value = int(max_value)
                # print('PGM type: {}'.format(b_type.decode('utf-8')))
                if b_type:
                    img_depth = 1
                else:
                    img_depth = 4
            
            elif file_type == 'raw' or file_type == 'RAW':
                raw_data = f.read()
                print(len(raw_data))
                assert len(raw_data) == 256*256
                f.seek(0)
                img_width, img_height, img_depth = 256, 256, 1

            col = []
            for _ in range(img_height):
                row = []
                for _ in range(img_width):
                    tmp = f.read(1 * img_depth)
                    dot = ord(tmp) if b_type else int(tmp)
                    row.append(dot)
                col.append(row)
            result = np.array(col, dtype=np.uint8)
            # print(result.shape)
        return result
    
    def add2Images(self):
        self.viewMode = 3
        fname = self.loadFile()
        im = self.pgm_processing(fname[0]) if fname[0].split('.')[-1] in ['pgm', 'ppm', 'raw', 'PGM', 'PPM', 'RAW'] else Image.open(fname[0])
        self.inImg = im if isinstance(im, np.ndarray) else np.array(im, dtype=np.uint8)

        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        # print(w, h)

        self.setviewMode(w,h)
        qImg = QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_Grayscale8) if self.inImg.shape[-1] != 3 else QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_RGB888)
        self.inImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
        fname2 = self.loadFile()
        im2 = self.pgm_processing(fname2[0]) if fname2[0].split('.')[-1] in ['pgm', 'ppm', 'raw', 'PGM', 'PPM', 'RAW'] else Image.open(fname2[0])
        self.inImg2 = im2 if isinstance(im2, np.ndarray) else np.array(im2, dtype=np.uint8)
        
        if self.inImg.shape != self.inImg2.shape: 
            self.inImg2 = nn_interpolate(self.inImg2, self.inImg.shape)
        
        qImg2 = QImage(self.inImg2.data, w, h, int(self.inImg2.nbytes / h), QImage.Format_Grayscale8) if self.inImg2.shape[-1] != 3 else QImage(self.inImg2.data, w, h, int(self.inImg2.nbytes / h), QImage.Format_RGB888)
        self.inImgLabel2.setPixmap(QPixmap.fromImage(qImg2))
        self.outImg = self.inImg.copy()
        
        alpha, beta = 1, 1
        for j in range(h):
            for i in range(w):
                val = int((alpha * self.inImg[j][i]) + (beta * self.inImg2[j][i]))
                self.outImg[j][i] = 255 if val > 255 else val
                
        qImg3 = QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_Grayscale8) if self.outImg.shape[-1] != 3 else QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_RGB888)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg3))
        
    def sub2Images(self):
        self.viewMode = 3
        fname = self.loadFile()
        im = self.pgm_processing(fname[0]) if fname[0].split('.')[-1] in ['pgm', 'ppm', 'raw', 'PGM', 'PPM', 'RAW'] else Image.open(fname[0])
        self.inImg = im if isinstance(im, np.ndarray) else np.array(im, dtype=np.uint8)

        w = self.inImg.shape[1]
        h = self.inImg.shape[0]
        
        self.setviewMode(w,h)
        qImg = QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_Grayscale8) if self.inImg.shape[-1] != 3 else QImage(self.inImg.data, w, h, int(self.inImg.nbytes/h), QImage.Format_RGB888)
        self.inImgLabel.setPixmap(QPixmap.fromImage(qImg))
        
        fname2 = self.loadFile()
        im2 = self.pgm_processing(fname2[0]) if fname2[0].split('.')[-1] in ['pgm', 'ppm', 'raw', 'PGM', 'PPM', 'RAW'] else Image.open(fname2[0])
        self.inImg2 = im2 if isinstance(im2, np.ndarray) else np.array(im2, dtype=np.uint8)
        
        if self.inImg.shape != self.inImg2.shape: 
            self.inImg2 = nn_interpolate(self.inImg2, self.inImg.shape)
        
        qImg2 = QImage(self.inImg2.data, w, h, int(self.inImg2.nbytes / h), QImage.Format_Grayscale8) if self.inImg2.shape[-1] != 3 else QImage(self.inImg2.data, w, h, int(self.inImg2.nbytes / h), QImage.Format_RGB888)
        self.inImgLabel2.setPixmap(QPixmap.fromImage(qImg2))
        self.outImg = self.inImg.copy()
        # import matplotlib.pyplot as plt
        # plt.imshow(self.outImg)
        # plt.show()
        alpha, beta = 1, 1
        for j in range(h):
            for i in range(w):
                val = int((alpha * self.inImg[j][i]) - (beta * self.inImg2[j][i]))
                self.outImg[j][i] = 0 if val < 0 else val
        # print(self.inImg.shape, self.inImg2.shape, self.outImg.shape)
        # plt.imshow(self.outImg)
        # plt.show()
        
        qImg3 = QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_Grayscale8) if self.outImg.shape[-1] != 3 else QImage(self.outImg.data, w, h, int(self.outImg.nbytes/h), QImage.Format_RGB888)
        self.outImgLabel.setPixmap(QPixmap.fromImage(qImg3))
        
if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        ex = ImageProcessing()
        sys.exit(app.exec_())
    except:
        pass
    