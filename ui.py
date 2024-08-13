from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QPushButton,
    QScrollArea,
    QLabel,
    QDialog,
    QGraphicsView,
    QGraphicsScene,
    QFrame,
    QGridLayout,
    
)

from PyQt5.QtGui import QIcon, QPixmap, QPainter, QBrush, QPen, QMouseEvent, QHoverEvent, QFont, QColor,QPalette
from PyQt5.QtCore import Qt, QLine, QPointF, QRectF, QLine, QEvent, QPropertyAnimation, pyqtProperty
import sqlite3
import random
import sys
import time

activationslist = ["",
                   "Tanh",
                   "ReLU",
                   "Sigmoid",
                   "Softmax_CrossEntropy",
                   "Softmax"]

lossfunclist = ["",
                "mean_square_error",
                "cross_entropy_loss",
                "",]

#comboboxcolor = "QComboBox""{""background-color: #5b5b5b;""}"

class Main_window(QMainWindow):
    
    """ User interface
    """
    
    def __init__(self):
        super(Main_window, self).__init__()
        
              
        
        
        
        self.setObjectName("MainWindow")
        self.resize(1099, 869)
        self.setWindowTitle("MainWindow")
        #self.setStyleSheet("QWidget { background-color: black}")
        
        #### Network definition ####
        self.network_frame = QtWidgets.QFrame(self)
        self.network_frame.setGeometry(QtCore.QRect(30, 110, 351, 691))
        self.network_frame.setFrameShape(QtWidgets.QFrame.Box)
        self.network_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.network_frame.setLineWidth(2)
        self.network_frame.setObjectName("network_frame")
        self.label = QtWidgets.QLabel(self.network_frame)
        self.label.setGeometry(QtCore.QRect(10, 10, 171, 16))
        self.label.setObjectName("label")
        
        ## Buttons Network ##
        self.pushButton_ntcreate = QtWidgets.QPushButton(self.network_frame)
        self.pushButton_ntcreate.setGeometry(QtCore.QRect(10, 650, 181, 31))
        self.pushButton_ntcreate.setObjectName("pushButton_ntcreate")
        #self.pushButton_ntcreate.clicked.connect(self.networkexporter)
        
        self.pushButton_ntclean = QtWidgets.QPushButton(self.network_frame)
        self.pushButton_ntclean.setGeometry(QtCore.QRect(190, 650, 151, 31))
        self.pushButton_ntclean.setObjectName("pushButton_ntclean")
        
        
        
        
        ## Neuron 1 ##
        self.frame_neuron1 = QtWidgets.QFrame(self.network_frame)
        self.frame_neuron1.setGeometry(QtCore.QRect(10, 30, 331, 71))
        self.frame_neuron1.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_neuron1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_neuron1.setObjectName("frame_neuron1")
        
        self.pushButton_n1 = QtWidgets.QPushButton(self.frame_neuron1)
        self.pushButton_n1.setGeometry(QtCore.QRect(300, 40, 21, 21))
        self.pushButton_n1.setObjectName("pushButton_n1")
        
        self.spinBox_n1 = QtWidgets.QSpinBox(self.frame_neuron1)
        self.spinBox_n1.setGeometry(QtCore.QRect(120, 10, 61, 22))
        self.spinBox_n1.setObjectName("spinBox_n1")
        
        self.comboBox_n1 = QtWidgets.QComboBox(self.frame_neuron1)
        self.comboBox_n1.setGeometry(QtCore.QRect(10, 40, 211, 22))
        self.comboBox_n1.setObjectName("comboBox_n1")
        self.comboBox_n1.addItems(activationslist)
        #self.comboBox_n1.setStyleSheet(comboboxcolor) 
        
        self.lineEdit_n1 = QtWidgets.QLineEdit(self.frame_neuron1)
        self.lineEdit_n1.setGeometry(QtCore.QRect(10, 10, 101, 20))
        self.lineEdit_n1.setObjectName("lineEdit_n1")
        
        self.checkBox_n1 = QtWidgets.QCheckBox(self.frame_neuron1)
        self.checkBox_n1.setGeometry(QtCore.QRect(230, 40, 61, 17))
        self.checkBox_n1.setObjectName("checkBox_n1")
        self.checkBox_n1.stateChanged.connect(self.chbox_n1)
        
        
        
        self.checkBox_conv_1 = QtWidgets.QCheckBox(self.frame_neuron1)
        self.checkBox_conv_1.setGeometry(QtCore.QRect(190, 10, 70, 17))
        self.checkBox_conv_1.setObjectName("checkBox_conv_1")
        
        self.checkBox_nkernell_1 = QtWidgets.QCheckBox(self.frame_neuron1)
        self.checkBox_nkernell_1.setGeometry(QtCore.QRect(260, 10, 70, 17))
        self.checkBox_nkernell_1.setObjectName("checkBox_nkernell_1")
        
        
        
        
                
        ## Neuron 2 ##
        self.frame_neuron2 = QtWidgets.QFrame(self.network_frame)
        self.frame_neuron2.setGeometry(QtCore.QRect(10, 100, 331, 71))
        self.frame_neuron2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_neuron2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_neuron2.setObjectName("frame_neuron2")
        self.frame_neuron2.setEnabled(False)
        
        self.pushButton_n2 = QtWidgets.QPushButton(self.frame_neuron2)
        self.pushButton_n2.setGeometry(QtCore.QRect(300, 40, 21, 21))
        self.pushButton_n2.setObjectName("pushButton_n2")
        
        self.spinBox_n2 = QtWidgets.QSpinBox(self.frame_neuron2)
        self.spinBox_n2.setGeometry(QtCore.QRect(120, 10, 61, 22))
        self.spinBox_n2.setObjectName("spinBox_n2")
        
        self.comboBox_n2 = QtWidgets.QComboBox(self.frame_neuron2)
        self.comboBox_n2.setGeometry(QtCore.QRect(10, 40, 211, 22))
        self.comboBox_n2.setObjectName("comboBox_n2")
        self.comboBox_n2.addItems(activationslist)
        #self.comboBox_n2.setStyleSheet(comboboxcolor) 
        
        self.lineEdit_n2 = QtWidgets.QLineEdit(self.frame_neuron2)
        self.lineEdit_n2.setGeometry(QtCore.QRect(10, 10, 101, 20))
        self.lineEdit_n2.setObjectName("lineEdit_n2")
        
        self.checkBox_n2 = QtWidgets.QCheckBox(self.frame_neuron2)
        self.checkBox_n2.setGeometry(QtCore.QRect(230, 40, 61, 17))
        self.checkBox_n2.setObjectName("checkBox_n2")
        self.checkBox_n2.stateChanged.connect(self.chbox_n2)
        
        self.checkBox_nkernell_2 = QtWidgets.QCheckBox(self.frame_neuron2)
        self.checkBox_nkernell_2.setGeometry(QtCore.QRect(260, 10, 70, 17))
        self.checkBox_nkernell_2.setObjectName("checkBox_nkernell_2")
        
        self.checkBox_conv_2 = QtWidgets.QCheckBox(self.frame_neuron2)
        self.checkBox_conv_2.setGeometry(QtCore.QRect(190, 10, 70, 17))
        self.checkBox_conv_2.setObjectName("checkBox_conv_2")
        
        
        ## Neuron 3 ##
        self.frame_neuron3 = QtWidgets.QFrame(self.network_frame)
        self.frame_neuron3.setGeometry(QtCore.QRect(10, 170, 331, 71))
        self.frame_neuron3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_neuron3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_neuron3.setObjectName("frame_neuron3")
        self.frame_neuron3.setEnabled(False)
        
        self.pushButton_n3 = QtWidgets.QPushButton(self.frame_neuron3)
        self.pushButton_n3.setGeometry(QtCore.QRect(300, 40, 21, 21))
        self.pushButton_n3.setObjectName("pushButton_n3")
        
        self.spinBox_n3 = QtWidgets.QSpinBox(self.frame_neuron3)
        self.spinBox_n3.setGeometry(QtCore.QRect(120, 10, 61, 22))
        self.spinBox_n3.setObjectName("spinBox_n3")
        
        self.comboBox_n3 = QtWidgets.QComboBox(self.frame_neuron3)
        self.comboBox_n3.setGeometry(QtCore.QRect(10, 40, 211, 22))
        self.comboBox_n3.setObjectName("comboBox_n3")
        self.comboBox_n3.addItems(activationslist)
        #self.comboBox_n3.setStyleSheet(comboboxcolor) 
        
        self.lineEdit_n3 = QtWidgets.QLineEdit(self.frame_neuron3)
        self.lineEdit_n3.setGeometry(QtCore.QRect(10, 10, 101, 20))
        self.lineEdit_n3.setObjectName("lineEdit_n3")
        
        self.checkBox_n3 = QtWidgets.QCheckBox(self.frame_neuron3)
        self.checkBox_n3.setGeometry(QtCore.QRect(230, 40, 61, 17))
        self.checkBox_n3.setObjectName("checkBox_n3")
        self.checkBox_n3.stateChanged.connect(self.chbox_n3)
        
        self.checkBox_nkernell_3 = QtWidgets.QCheckBox(self.frame_neuron3)
        self.checkBox_nkernell_3.setGeometry(QtCore.QRect(260, 10, 70, 17))
        self.checkBox_nkernell_3.setObjectName("checkBox_nkernell_3")
        
        self.checkBox_conv_3 = QtWidgets.QCheckBox(self.frame_neuron3)
        self.checkBox_conv_3.setGeometry(QtCore.QRect(190, 10, 70, 17))
        self.checkBox_conv_3.setObjectName("checkBox_conv_3")
        
        ## Neuron 4 ##
        self.frame_neuron4 = QtWidgets.QFrame(self.network_frame)
        self.frame_neuron4.setGeometry(QtCore.QRect(10, 240, 331, 71))
        self.frame_neuron4.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_neuron4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_neuron4.setObjectName("frame_neuron4")
        self.frame_neuron4.setEnabled(False)
        
        self.pushButton_n4 = QtWidgets.QPushButton(self.frame_neuron4)
        self.pushButton_n4.setGeometry(QtCore.QRect(300, 40, 21, 21))
        self.pushButton_n4.setObjectName("pushButton_n4")
        
        self.spinBox_n4 = QtWidgets.QSpinBox(self.frame_neuron4)
        self.spinBox_n4.setGeometry(QtCore.QRect(120, 10, 61, 22))
        self.spinBox_n4.setObjectName("spinBox_n4")
        
        self.comboBox_n4 = QtWidgets.QComboBox(self.frame_neuron4)
        self.comboBox_n4.setGeometry(QtCore.QRect(10, 40, 211, 22))
        self.comboBox_n4.setObjectName("comboBox_n4")
        self.comboBox_n4.addItems(activationslist)
        #self.comboBox_n4.setStyleSheet(comboboxcolor) 
        
        self.lineEdit_n4 = QtWidgets.QLineEdit(self.frame_neuron4)
        self.lineEdit_n4.setGeometry(QtCore.QRect(10, 10, 101, 20))
        self.lineEdit_n4.setObjectName("lineEdit_n4")
        
        self.checkBox_n4 = QtWidgets.QCheckBox(self.frame_neuron4)
        self.checkBox_n4.setGeometry(QtCore.QRect(230, 40, 61, 17))
        self.checkBox_n4.setObjectName("checkBox_n4")
        self.checkBox_n4.stateChanged.connect(self.chbox_n4)
        
        
        self.checkBox_nkernell_4 = QtWidgets.QCheckBox(self.frame_neuron4)
        self.checkBox_nkernell_4.setGeometry(QtCore.QRect(260, 10, 70, 17))
        self.checkBox_nkernell_4.setObjectName("checkBox_nkernell_4")
        
        self.checkBox_conv_4 = QtWidgets.QCheckBox(self.frame_neuron4)
        self.checkBox_conv_4.setGeometry(QtCore.QRect(190, 10, 70, 17))
        self.checkBox_conv_4.setObjectName("checkBox_conv_4")
        
        
        ## Neuron 5 ##
        self.frame_neuron5 = QtWidgets.QFrame(self.network_frame)
        self.frame_neuron5.setGeometry(QtCore.QRect(10, 310, 331, 71))
        self.frame_neuron5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_neuron5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_neuron5.setObjectName("frame_neuron5")
        self.frame_neuron5.setEnabled(False)
        
        
        self.pushButton_n5 = QtWidgets.QPushButton(self.frame_neuron5)
        self.pushButton_n5.setGeometry(QtCore.QRect(300, 40, 21, 21))
        self.pushButton_n5.setObjectName("pushButton_n5")
        
        self.spinBox_n5 = QtWidgets.QSpinBox(self.frame_neuron5)
        self.spinBox_n5.setGeometry(QtCore.QRect(120, 10, 61, 22))
        self.spinBox_n5.setObjectName("spinBox_n5")
        
        self.comboBox_n5 = QtWidgets.QComboBox(self.frame_neuron5)
        self.comboBox_n5.setGeometry(QtCore.QRect(10, 40, 211, 22))
        self.comboBox_n5.setObjectName("comboBox_n5")
        self.comboBox_n5.addItems(activationslist)
        #self.comboBox_n5.setStyleSheet(comboboxcolor) 
        
        self.lineEdit_n5 = QtWidgets.QLineEdit(self.frame_neuron5)
        self.lineEdit_n5.setGeometry(QtCore.QRect(10, 10, 101, 20))
        self.lineEdit_n5.setObjectName("lineEdit_n5")
        
        self.checkBox_n5 = QtWidgets.QCheckBox(self.frame_neuron5)
        self.checkBox_n5.setGeometry(QtCore.QRect(230, 40, 61, 17))
        self.checkBox_n5.setObjectName("checkBox_n5")
        self.checkBox_n5.stateChanged.connect(self.chbox_n5)
        
        self.checkBox_nkernell_5 = QtWidgets.QCheckBox(self.frame_neuron5)
        self.checkBox_nkernell_5.setGeometry(QtCore.QRect(260, 10, 70, 17))
        self.checkBox_nkernell_5.setObjectName("checkBox_nkernell_5")
        
        self.checkBox_conv_5 = QtWidgets.QCheckBox(self.frame_neuron5)
        self.checkBox_conv_5.setGeometry(QtCore.QRect(190, 10, 70, 17))
        self.checkBox_conv_5.setObjectName("checkBox_conv_5")
        
        ## Neuron 6 ##
        self.frame_neuron6 = QtWidgets.QFrame(self.network_frame)
        self.frame_neuron6.setGeometry(QtCore.QRect(10, 380, 331, 71))
        self.frame_neuron6.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_neuron6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_neuron6.setObjectName("frame_neuron6")
        self.frame_neuron6.setEnabled(False)
        
        
        self.pushButton_n6 = QtWidgets.QPushButton(self.frame_neuron6)
        self.pushButton_n6.setGeometry(QtCore.QRect(300, 40, 21, 21))
        self.pushButton_n6.setObjectName("pushButton_n6")
        
        self.spinBox_n6 = QtWidgets.QSpinBox(self.frame_neuron6)
        self.spinBox_n6.setGeometry(QtCore.QRect(120, 10, 61, 22))
        self.spinBox_n6.setObjectName("spinBox_n6")
        
        self.comboBox_n6 = QtWidgets.QComboBox(self.frame_neuron6)
        self.comboBox_n6.setGeometry(QtCore.QRect(10, 40, 211, 22))
        self.comboBox_n6.setObjectName("comboBox_n6")
        self.comboBox_n6.addItems(activationslist)
        #self.comboBox_n6.setStyleSheet(comboboxcolor) 
        
        self.lineEdit_n6 = QtWidgets.QLineEdit(self.frame_neuron6)
        self.lineEdit_n6.setGeometry(QtCore.QRect(10, 10, 101, 20))
        self.lineEdit_n6.setObjectName("lineEdit_n6")
        
        self.checkBox_n6 = QtWidgets.QCheckBox(self.frame_neuron6)
        self.checkBox_n6.setGeometry(QtCore.QRect(230, 40, 61, 17))
        self.checkBox_n6.setObjectName("checkBox_n6")
        self.checkBox_n6.stateChanged.connect(self.chbox_n6)
        
        self.checkBox_nkernell_6 = QtWidgets.QCheckBox(self.frame_neuron6)
        self.checkBox_nkernell_6.setGeometry(QtCore.QRect(260, 10, 70, 17))
        self.checkBox_nkernell_6.setObjectName("checkBox_nkernell_6")
        
        self.checkBox_conv_6 = QtWidgets.QCheckBox(self.frame_neuron6)
        self.checkBox_conv_6.setGeometry(QtCore.QRect(190, 10, 70, 17))
        self.checkBox_conv_6.setObjectName("checkBox_conv_6")
        
        ## Neuron 7 ##
        self.frame_neuron7 = QtWidgets.QFrame(self.network_frame)
        self.frame_neuron7.setGeometry(QtCore.QRect(10, 450, 331, 71))
        self.frame_neuron7.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_neuron7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_neuron7.setObjectName("frame_neuron7")
        self.frame_neuron7.setEnabled(False)
        
        
        self.pushButton_n7 = QtWidgets.QPushButton(self.frame_neuron7)
        self.pushButton_n7.setGeometry(QtCore.QRect(300, 40, 21, 21))
        self.pushButton_n7.setObjectName("pushButton_n7")
        
        self.spinBox_n7 = QtWidgets.QSpinBox(self.frame_neuron7)
        self.spinBox_n7.setGeometry(QtCore.QRect(120, 10, 61, 22))
        self.spinBox_n7.setObjectName("spinBox_n7")
        
        self.comboBox_n7 = QtWidgets.QComboBox(self.frame_neuron7)
        self.comboBox_n7.setGeometry(QtCore.QRect(10, 40, 211, 22))
        self.comboBox_n7.setObjectName("comboBox_n7")
        self.comboBox_n7.addItems(activationslist)
        #self.comboBox_n7.setStyleSheet(comboboxcolor) 
        
        self.lineEdit_n7 = QtWidgets.QLineEdit(self.frame_neuron7)
        self.lineEdit_n7.setGeometry(QtCore.QRect(10, 10, 101, 20))
        self.lineEdit_n7.setObjectName("lineEdit_n7")
        
        self.checkBox_n7 = QtWidgets.QCheckBox(self.frame_neuron7)
        self.checkBox_n7.setGeometry(QtCore.QRect(230, 40, 61, 17))
        self.checkBox_n7.setObjectName("checkBox_n7")
        self.checkBox_n7.stateChanged.connect(self.chbox_n7)
        
        self.checkBox_nkernell_7 = QtWidgets.QCheckBox(self.frame_neuron7)
        self.checkBox_nkernell_7.setGeometry(QtCore.QRect(260, 10, 70, 17))
        self.checkBox_nkernell_7.setObjectName("checkBox_nkernell_7")
        
        self.checkBox_conv_7 = QtWidgets.QCheckBox(self.frame_neuron7)
        self.checkBox_conv_7.setGeometry(QtCore.QRect(190, 10, 70, 17))
        self.checkBox_conv_7.setObjectName("checkBox_conv_7")
        
        ## Neuron 8 ##
        self.frame_neuron8 = QtWidgets.QFrame(self.network_frame)
        self.frame_neuron8.setEnabled(False)
        self.frame_neuron8.setGeometry(QtCore.QRect(10, 520, 331, 71))
        self.frame_neuron8.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_neuron8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_neuron8.setObjectName("frame_neuron8")
        
        self.pushButton_n8 = QtWidgets.QPushButton(self.frame_neuron8)
        self.pushButton_n8.setGeometry(QtCore.QRect(300, 40, 21, 21))
        self.pushButton_n8.setObjectName("pushButton_n8")
        
        self.spinBox_n8 = QtWidgets.QSpinBox(self.frame_neuron8)
        self.spinBox_n8.setGeometry(QtCore.QRect(120, 10, 61, 22))
        self.spinBox_n8.setObjectName("spinBox_n8")
        
        self.comboBox_n8 = QtWidgets.QComboBox(self.frame_neuron8)
        self.comboBox_n8.setGeometry(QtCore.QRect(10, 40, 211, 22))
        self.comboBox_n8.setObjectName("comboBox_n8")
        self.comboBox_n8.addItems(activationslist)
        #self.comboBox_n8.setStyleSheet(comboboxcolor) 
        
        self.lineEdit_n8 = QtWidgets.QLineEdit(self.frame_neuron8)
        self.lineEdit_n8.setGeometry(QtCore.QRect(10, 10, 101, 20))
        self.lineEdit_n8.setObjectName("lineEdit_n8")
        
        self.checkBox_n8 = QtWidgets.QCheckBox(self.frame_neuron8)
        self.checkBox_n8.setGeometry(QtCore.QRect(230, 40, 61, 17))
        self.checkBox_n8.setObjectName("checkBox_n8")
        
        self.checkBox_nkernell_8 = QtWidgets.QCheckBox(self.frame_neuron8)
        self.checkBox_nkernell_8.setGeometry(QtCore.QRect(260, 10, 70, 17))
        self.checkBox_nkernell_8.setObjectName("checkBox_nkernell_8")
        
        self.checkBox_conv_8 = QtWidgets.QCheckBox(self.frame_neuron8)
        self.checkBox_conv_8.setGeometry(QtCore.QRect(190, 10, 70, 17))
        self.checkBox_conv_8.setObjectName("checkBox_conv_8")
        
        
        ## Loss ##
        self.frame_loss = QtWidgets.QFrame(self.network_frame)
        self.frame_loss.setGeometry(QtCore.QRect(10, 590, 331, 61))
        self.frame_loss.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_loss.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_loss.setLineWidth(2)
        self.frame_loss.setObjectName("frame_loss")
        
        self.loss_label = QtWidgets.QLabel(self.frame_loss)
        self.loss_label.setGeometry(QtCore.QRect(10, 20, 81, 16))
        self.loss_label.setObjectName("loss_label")
        
        self.comboBox = QtWidgets.QComboBox(self.frame_loss)
        self.comboBox.setGeometry(QtCore.QRect(100, 20, 211, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(lossfunclist)
        #self.comboBox.setStyleSheet(comboboxcolor) 
        
        
        
        self.listofneurons = [self.frame_neuron1,
                              self.frame_neuron2,
                              self.frame_neuron3,
                              self.frame_neuron4,
                              self.frame_neuron5,
                              self.frame_neuron6,
                              self.frame_neuron7,
                              self.frame_neuron8,]
        
        self.listofchecks = [self.checkBox_n1,
                             self.checkBox_n2,
                             self.checkBox_n3,
                             self.checkBox_n4,
                             self.checkBox_n5,
                             self.checkBox_n6,
                             self.checkBox_n7,
                             self.checkBox_n8,]
        
        self.listofnumbers = [self.spinBox_n1,
                              self.spinBox_n2,
                              self.spinBox_n3,
                              self.spinBox_n4,
                              self.spinBox_n5,
                              self.spinBox_n6,
                              self.spinBox_n7,
                              self.spinBox_n8,]
        
        self.activlist = [self.comboBox_n1,
                          self.comboBox_n2,
                          self.comboBox_n3,
                          self.comboBox_n4,
                          self.comboBox_n5,
                          self.comboBox_n6,
                          self.comboBox_n7,
                          self.comboBox_n8,]
        
        #### Training ####
        self.training_frame = QtWidgets.QFrame(self)
        self.training_frame.setGeometry(QtCore.QRect(390, 270, 691, 251))
        self.training_frame.setFrameShape(QtWidgets.QFrame.Box)
        self.training_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.training_frame.setLineWidth(2)
        self.training_frame.setObjectName("training_frame")
        
        self.progressBar_epoch = QtWidgets.QProgressBar(self.training_frame)
        self.progressBar_epoch.setGeometry(QtCore.QRect(30, 110, 641, 23))
        self.progressBar_epoch.setProperty("value", 24)
        self.progressBar_epoch.setObjectName("progressBar_epoch")
        
        self.pushButton = QtWidgets.QPushButton(self.training_frame)
        self.pushButton.setGeometry(QtCore.QRect(730, 100, 75, 23))
        self.pushButton.setObjectName("pushButton")
        
        self.pushButton_starttrain = QtWidgets.QPushButton(self.training_frame)
        self.pushButton_starttrain.setGeometry(QtCore.QRect(554, 20, 121, 41))
        self.pushButton_starttrain.setObjectName("pushButton_starttrain")
        
        self.progressBar_batch = QtWidgets.QProgressBar(self.training_frame)
        self.progressBar_batch.setGeometry(QtCore.QRect(30, 180, 641, 23))
        self.progressBar_batch.setProperty("value", 24)
        self.progressBar_batch.setObjectName("progressBar_batch")
        
        self.pushButton_canceltrain = QtWidgets.QPushButton(self.training_frame)
        self.pushButton_canceltrain.setGeometry(QtCore.QRect(600, 220, 75, 23))
        self.pushButton_canceltrain.setObjectName("pushButton_canceltrain")
        
        self.checkBox_testontheway = QtWidgets.QCheckBox(self.training_frame)
        self.checkBox_testontheway.setGeometry(QtCore.QRect(40, 220, 311, 17))
        self.checkBox_testontheway.setObjectName("checkBox_testontheway")
        
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.training_frame)
        self.doubleSpinBox.setGeometry(QtCore.QRect(450, 30, 62, 22))
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        
        self.spinBox = QtWidgets.QSpinBox(self.training_frame)
        self.spinBox.setGeometry(QtCore.QRect(120, 30, 71, 22))
        self.spinBox.setObjectName("spinBox")
        
        self.label_epochs = QtWidgets.QLabel(self.training_frame)
        self.label_epochs.setGeometry(QtCore.QRect(69, 30, 47, 13))
        self.label_epochs.setObjectName("label_epochs")
        
        self.label_learnig = QtWidgets.QLabel(self.training_frame)
        self.label_learnig.setGeometry(QtCore.QRect(340, 30, 81, 16))
        self.label_learnig.setObjectName("label_learnig")
        
        self.label_advanceepoch = QtWidgets.QLabel(self.training_frame)
        self.label_advanceepoch.setGeometry(QtCore.QRect(30, 70, 91, 20))
        self.label_advanceepoch.setObjectName("label_advanceepoch")
        
        self.label_advancetraining = QtWidgets.QLabel(self.training_frame)
        self.label_advancetraining.setGeometry(QtCore.QRect(30, 150, 101, 16))
        self.label_advancetraining.setObjectName("label_advancetraining")
        
        
        #### Database ####
        self.database_frame = QtWidgets.QFrame(self)
        self.database_frame.setGeometry(QtCore.QRect(390, 111, 691, 151))
        self.database_frame.setFrameShape(QtWidgets.QFrame.Box)
        self.database_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.database_frame.setLineWidth(2)
        self.database_frame.setObjectName("database_frame")
        
        self.label_4 = QtWidgets.QLabel(self.database_frame)
        self.label_4.setGeometry(QtCore.QRect(20, 20, 301, 16))
        self.label_4.setObjectName("label_4")
        
        self.pushButton_2 = QtWidgets.QPushButton(self.database_frame)
        self.pushButton_2.setGeometry(QtCore.QRect(580, 110, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        
        
        #### Mathplotlib ####
        self.frame_mathplotlib = QtWidgets.QFrame(self)
        self.frame_mathplotlib.setGeometry(QtCore.QRect(820, 530, 261, 271))
        self.frame_mathplotlib.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_mathplotlib.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_mathplotlib.setLineWidth(2)
        self.frame_mathplotlib.setObjectName("frame_mathplotlib")
        self.label_2 = QtWidgets.QLabel(self.frame_mathplotlib)
        self.label_2.setGeometry(QtCore.QRect(30, 10, 201, 16))
        self.label_2.setObjectName("label_2")
        
        
        #### Results ####
        self.frame_results = QtWidgets.QFrame(self)
        self.frame_results.setGeometry(QtCore.QRect(390, 531, 421, 271))
        self.frame_results.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_results.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_results.setLineWidth(2)
        self.frame_results.setObjectName("frame_results")
        self.label_3 = QtWidgets.QLabel(self.frame_results)
        self.label_3.setGeometry(QtCore.QRect(20, 10, 261, 16))
        self.label_3.setObjectName("label_3")
        
        self.label.setText("Area de definicion de la red")
        self.pushButton_ntclean.setText("PushButton")
        self.loss_label.setText("Loss Function")
        self.pushButton_n1.setText( "PushButton cancelation of neuron")
        self.lineEdit_n1.setInputMask("neuron")
        self.lineEdit_n1.setText( "neuron")
        self.checkBox_n1.setText("CheckBox")
        self.checkBox_conv_1.setText("CheckBox")
        self.checkBox_nkernell_1.setText("CheckBox")
        self.pushButton_n2.setText( "PushButton cancelation of neuron")
        self.lineEdit_n2.setInputMask( "neuron")
        self.lineEdit_n2.setText( "neuron")
        self.checkBox_n2.setText("CheckBox")
        self.checkBox_nkernell_2.setText( "CheckBox")
        self.checkBox_conv_2.setText( "CheckBox")
        
        self.pushButton_n3.setText( "PushButton cancelation of neuron")
        self.lineEdit_n3.setInputMask( "neuron")
        self.lineEdit_n3.setText( "neuron")
        self.checkBox_n3.setText( "CheckBox")
        self.checkBox_nkernell_3.setText( "CheckBox")
        self.checkBox_conv_3.setText( "CheckBox")
        self.pushButton_n4.setText( "PushButton cancelation of neuron")
        self.lineEdit_n4.setInputMask( "neuron")
        self.lineEdit_n4.setText( "neuron")
        self.checkBox_n4.setText( "CheckBox")
        self.checkBox_nkernell_4.setText( "CheckBox")
        self.checkBox_conv_4.setText( "CheckBox")
        self.pushButton_n5.setText( "PushButton cancelation of neuron")
        self.lineEdit_n5.setInputMask( "neuron")
        self.lineEdit_n5.setText( "neuron")
        self.checkBox_n5.setText( "CheckBox")
        self.checkBox_nkernell_5.setText( "CheckBox")
        self.checkBox_conv_5.setText( "CheckBox")
        self.pushButton_n6.setText( "PushButton cancelation of neuron")
        self.lineEdit_n6.setInputMask( "neuron")
        self.lineEdit_n6.setText( "neuron")
        self.checkBox_n6.setText( "CheckBox")
        self.checkBox_nkernell_6.setText( "CheckBox")
        self.checkBox_conv_6.setText( "CheckBox")
        self.pushButton_n7.setText( "PushButton cancelation of neuron")
        self.lineEdit_n7.setInputMask( "neuron")
        self.lineEdit_n7.setText( "neuron")
        self.checkBox_n7.setText( "CheckBox")
        self.checkBox_nkernell_7.setText( "CheckBox")
        self.checkBox_conv_7.setText( "CheckBox")
        self.pushButton_n8.setText( "PushButton cancelation of neuron")
        self.lineEdit_n8.setInputMask( "neuron")
        self.lineEdit_n8.setText( "neuron")
        self.checkBox_n8.setText( "CheckBox")
        self.checkBox_nkernell_8.setText( "CheckBox")
        self.checkBox_conv_8.setText( "CheckBox")
        self.pushButton_ntcreate.setText( "Create Network")
        self.pushButton.setText( "PushButton")
        self.pushButton_starttrain.setText( "Start Training/Testing")
        self.pushButton_canceltrain.setText( "PushButton")
        self.checkBox_testontheway.setText( "CheckBox para indicar testeo")
        self.label_epochs.setText( "Epochs:")
        self.label_learnig.setText( "Learning Rate:")
        self.label_advanceepoch.setText( "Advance on Epoch:")
        self.label_advancetraining.setText( "Advance on Training:")
        self.label_4.setText( "TextLabel carga de databases")
        self.pushButton_2.setText( "PushButton")
        self.label_2.setText( "TextLabel quitar: va matpltlib")
        self.label_3.setText( "TextLabel Arrea de resultados")
        
        
    def chbox_n1(self):
            if self.checkBox_n1.isChecked():
                self.frame_neuron2.setEnabled(True)
            else:
                self.frame_neuron2.setEnabled(False)
                self.frame_neuron3.setEnabled(False)
                self.frame_neuron4.setEnabled(False)
                self.frame_neuron5.setEnabled(False)
                self.frame_neuron6.setEnabled(False)
                self.frame_neuron7.setEnabled(False)
                self.frame_neuron8.setEnabled(False)
                
                
    def chbox_n2(self):
            if self.checkBox_n2.isChecked():
                self.frame_neuron3.setEnabled(True)
            else:
                self.frame_neuron3.setEnabled(False)
                self.frame_neuron4.setEnabled(False)
                self.frame_neuron5.setEnabled(False)
                self.frame_neuron6.setEnabled(False)
                self.frame_neuron7.setEnabled(False)
                self.frame_neuron8.setEnabled(False)
    
    def chbox_n3(self):
            if self.checkBox_n3.isChecked():
                self.frame_neuron4.setEnabled(True)
            else:
                self.frame_neuron4.setEnabled(False)
                self.frame_neuron5.setEnabled(False)
                self.frame_neuron6.setEnabled(False)
                self.frame_neuron7.setEnabled(False)
                self.frame_neuron8.setEnabled(False)
    
    def chbox_n4(self):
            if self.checkBox_n4.isChecked():
                self.frame_neuron5.setEnabled(True)
            else:
                self.frame_neuron5.setEnabled(False)
                self.frame_neuron6.setEnabled(False)
                self.frame_neuron7.setEnabled(False)
                self.frame_neuron8.setEnabled(False)
    
    def chbox_n5(self):
            if self.checkBox_n5.isChecked():
                self.frame_neuron6.setEnabled(True)
            else:
                self.frame_neuron6.setEnabled(False)
                self.frame_neuron7.setEnabled(False)
                self.frame_neuron8.setEnabled(False)
    
    def chbox_n6(self):
            if self.checkBox_n6.isChecked():
                self.frame_neuron7.setEnabled(True)
            else:
                self.frame_neuron7.setEnabled(False)
                self.frame_neuron8.setEnabled(False)
    
    def chbox_n7(self):
            if self.checkBox_n7.isChecked():
                self.frame_neuron8.setEnabled(True)
            else:
                self.frame_neuron8.setEnabled(False)
                
    
                
        
        
        

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    
    ui = Main_window()
       
    

    ui.show()

    sys.exit(app.exec_())