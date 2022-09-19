import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

class ButtonWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        btn1 = QPushButton('Button1', self)
        
        btn2 = QPushButton(self)
        btn2.setText('Close')
        
        btn1.move(20,20)
        btn2.move(100,20)
        
        self.setWindowTitle('QPushButton')
        self.setGeometry(300,300,200,60)
        self.show()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ButtonWindow()
    sys.exit(app.exec_())
    