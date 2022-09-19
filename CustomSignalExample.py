import sys
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal

class SendTextWidget(QWidget):
    command = pyqtSignal(str)
    
    def __init__(self, parent) -> None:
        super(SendTextWidget, self).__init__(parent, Qt.Window)
        self.le = QLineEdit('Text')
        self.bt = QPushButton('Send')
        self.initUI()
        
    def initUI(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.le)
        vbox.addWidget(self.bt)
        self.le.setAlignment(Qt.AlignCenter)
        vbox.setAlignment(Qt.AlignCenter)
        self.setLayout(vbox)
        
        self.bt.clicked.connect(self.btClicked)
        
        self.setWindowTitle('SendTextWidget')
        self.setGeometry(500,300,200,200)
        
        self.show()
        
    def btClicked(self):
        self.command.emit(self.le.text())
        
class ReceiveTextWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.lb = QLabel('Text')
        self.initUI()
        
    def initUI(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.lb)
        vbox.setAlignment(Qt.AlignCenter)
        self.setLayout(vbox)
        
        self.setWindowTitle('ReceiveTextWidget')
        self.setGeometry(300,300,200,200)
        
        s = SendTextWidget(self)
        s.command.connect(self.changeText)
        
        self.show()
        
    def changeText(self, str):
        self.lb.setText(str)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    r = ReceiveTextWidget()
    sys.exit(app.exec_())
    