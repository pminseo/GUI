import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

class ButtonWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def __del__(self):
        print("Button Window Closed")    
    
    def initUI(self):
        btn1 = QPushButton('Button1', self)
        
        btn2 = QPushButton(self)
        btn2.setText('Close')
        
        btn1.move(20,20)
        btn2.move(100,20)
        
        btn2.clicked.connect(self.closeButtonClicked)
        
        self.setWindowTitle('QPushButton')
        self.setGeometry(300,300,200,60)
        self.show()
    
    def closeButtonClicked(self):
        print("Close Button Clicked")
        self.close()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ButtonWindow()
    sys.exit(app.exec_())
    