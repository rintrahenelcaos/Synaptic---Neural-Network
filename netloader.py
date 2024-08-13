from net import Net_Propper, Net_Proper2
import numpy as np

import sys

import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical   

import matplotlib as mpl
import matplotlib.pyplot as plt

import threading

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from activations import Activation_Layer, Softmax, Softmax_CrossEntropy, Sigmoid, Tanh,ReLU
from activations import cross_entropy_loss, cross_entropy_loss_der, cross_entropy_loss_deriv, softmax_crossentropy_der, mean_square_error, mean_square_error_der
from view import Ui_MainWindow



data = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

trainer_labels = (np.eye(10)[train_labels]).T
tester_labels = (np.eye(10)[test_labels]).T

train_images = train_images/255
test_images = test_images/255

pruebareshape = train_images[0]

trainer_images = train_images.reshape(60000,784).T
tester_images = test_images.reshape(10000, 784).T

tester_imagescorto = tester_images[:,0:2]
tester_labelscorto = tester_labels[:,0:2]
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
Y2 = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]).T
batch = tuple (zip(X, Y))
errors=[]


"""app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
"""

network = Net_Propper

class Trainer(QObject):
    
    trained = pyqtSignal()
    
    def networktrainthread(self, trainer_images, trainer_labels, neuronclip, activationsclip, epochs , learning_rate, lossfunclip, losscalder):
        
        self.network = Net_Propper(trainer_images, trainer_labels, neuronclip, activationsclip, epochs , learning_rate, lossfunclip, losscalder)
        self.network.control_function()
        self.traindewei, self.trainedbi = self.network.starttrain(True, True)
        self.trained.emit()
    
    #def trainerthred(self):
        
        
        



class NetLoader():
    def __init__(self):
        #app = QtWidgets.QApplication(sys.argv)
        self.MainWindow = QtWidgets.QMainWindow()
        self.visualizer = Ui_MainWindow()
        self.visualizer.setupUi(self.MainWindow)
        
        self.visualizer.pushButton_ntcreate.clicked.connect(lambda: self.neuroncharger())
        self.visualizer.pushButton_starttrain.clicked.connect(lambda: self.trainnetwork())
        self.visualizer.progressBar_batch.setValue(0)
        #sys.exit(app.exec_())
        self.listofneurons = [self.visualizer.frame_neuron1,
                              self.visualizer.frame_neuron2,
                              self.visualizer.frame_neuron3,
                              self.visualizer.frame_neuron4,
                              self.visualizer.frame_neuron5,
                              self.visualizer.frame_neuron6,
                              self.visualizer.frame_neuron7,
                              self.visualizer.frame_neuron8,]
        
        self.listofchecks = [self.visualizer.checkBox_n1,
                             self.visualizer.checkBox_n2,
                             self.visualizer.checkBox_n3,
                             self.visualizer.checkBox_n4,
                             self.visualizer.checkBox_n5,
                             self.visualizer.checkBox_n6,
                             self.visualizer.checkBox_n7,
                             self.visualizer.checkBox_n8,]
        
        self.listofnumbers = [self.visualizer.spinBox_n1,
                              self.visualizer.spinBox_n2,
                              self.visualizer.spinBox_n3,
                              self.visualizer.spinBox_n4,
                              self.visualizer.spinBox_n5,
                              self.visualizer.spinBox_n6,
                              self.visualizer.spinBox_n7,
                              self.visualizer.spinBox_n8,]
        
        self.activlist = [self.visualizer.comboBox_n1,
                          self.visualizer.comboBox_n2,
                          self.visualizer.comboBox_n3,
                          self.visualizer.comboBox_n4,
                          self.visualizer.comboBox_n5,
                          self.visualizer.comboBox_n6,
                          self.visualizer.comboBox_n7,
                          self.visualizer.comboBox_n8,]
        
        activationslist = ["Tanh",
                            "ReLU",
                            "Sigmoid",
                            "Softmax_CrossEntropy",
                            "Softmax"]
        listactivations = [Tanh, 
                           ReLU, 
                           Sigmoid, 
                           Softmax_CrossEntropy, 
                           Softmax]
        
        lossfunclist = ["mean_square_error",
                        "cross_entropy_loss"]
        listlossfunc = [mean_square_error, 
                        cross_entropy_loss]
        
        self.activationscharger = {activationslist: listactivations for activationslist,
                              listactivations in zip(activationslist, listactivations)}
        
        self.lossfunccharger = {lossfunclist: listlossfunc for lossfunclist,
                           listlossfunc in zip(lossfunclist, listlossfunc)}
        
                        
        
    """def neuroncharger(self):
        
        self.neuronsnumbers, self.activlist = self.visualizer.networkexporter()
        print(self.neuronsnumbers)
        print(self.activlist)"""
    
    def neuroncharger(self):
        self.activationslistexport = []
        self.neuronslist = []
        for i in range(len(self.listofneurons)):
            if self.listofneurons[i].isEnabled():
                if self.listofchecks[i].isChecked():
                    if self.activlist[i].currentText() != "":
                        self.neuronslist.append(int(self.listofnumbers[i].text()))
                        self.activationslistexport.append(self.activationscharger.get(self.activlist[i].currentText())())
        
        self.neuronslist.pop(-1)
            
        self.lossfunctionuse = self.lossfunccharger.get(self.visualizer.comboBox.currentText())
                
        print(self.neuronslist)
        
        print(self.neuronslist)
        print(self.activationslistexport)
        print(self.lossfunctionuse)          
        return self.neuronslist, self.activationslistexport, self.lossfunctionuse
    
    def trainerconfig(self):
        if self.lossfunctionuse == mean_square_error:
            self.lossfuncder = mean_square_error_der
        elif self.lossfunctionuse == cross_entropy_loss:
            if self.neuronslist[-1] == Softmax_CrossEntropy:
                self.lossfuncder = softmax_crossentropy_der
            else:
                self.lossfuncder = cross_entropy_loss_der
        return self.lossfuncder
        
    
    
    def networkcreator(self):
        neuronclip, activationsclip, lossfunclip = self.neuroncharger()
        self.lossfuncderclip = self.trainerconfig()
        
        
        self.network = Net_Propper(trainer_images, trainer_labels, neuronclip, activationsclip,int(self.visualizer.spinBox.text()) ,float(self.visualizer.doubleSpinBox.text().replace(",", ".")), lossfunclip, self.lossfuncderclip)
        self.network.control_function()
    
    def trainnetwork(self):
        
        self.networkcreator()
        self.visualizer.progressBar_batch.setValue(self.network.epochcounter) # not working
        self.visualizer.progressBar_epoch.setValue(self.network.lapmarc) # not working
        self.traindewei, self.trainedbi = self.network.starttrain(True, True)
        
        print(self.network.lapmarc)
        
        return self.traindewei, self.trainedbi
    
    def testnetwork(self):
        
        print(self.network.test(self.traindewei,self.trainedbi, tester_images, tester_labels, True))
    
    
    
    
        
        
        
#netter = Net_Proper()
#netter = Net_Proper2(trainer_images, trainer_labels, 10, 10, ReLU(), ReLU(),Softmax_CrossEntropy(), 1, 0.3, cross_entropy_loss, softmax_crossentropy_der)
#trainedparameters = netter.train3(shower= True, graph=False )
#print(len(trainedparameters))
#print("trainedparameters: [0]: ", trainedparameters[0].shape)
#weights1, biases1, weights2, biases2, weights3, biases3 = trainedparameters[0],trainedparameters[1],trainedparameters[2],trainedparameters[3], trainedparameters[4], trainedparameters[5]
#
#print(netter.test1(weights1, biases1,weights2, biases2, weights3, biases3, tester_images, tester_labels, True))
#netter.test4(trainedparameters, tester_images, tester_labels, True)
"""
print("con cargador")
neuronas = [10, 5]
activaciones = [ReLU(), Tanh(), Softmax_CrossEntropy()]
netter2 = Net_Propper(trainer_images, trainer_labels, neuronas,activaciones, 1, 0.7, cross_entropy_loss, softmax_crossentropy_der)
netter2.control_function()

traindewei, trainedbi = netter2.train(True, True)
"""
#netter2.shower()
"""print(netter2.test(traindewei,trainedbi, tester_images, tester_labels, True))"""






if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    gui = NetLoader()
    gui.MainWindow.show()
    sys.exit(app.exec_())
    
    #gui.big_bang()
    #print(gui.system_object_dic)
    #print("-"*20)
    #gui.great_weaver()
    




    
    















