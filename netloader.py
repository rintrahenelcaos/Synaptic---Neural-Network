from net import Net_Propper, Net_Proper2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical   

import matplotlib as mpl
import matplotlib.pyplot as plt
from activations import Activation_Layer, Softmax, Softmax_CrossEntropy, Sigmoid, Tanh,ReLU, cross_entropy_loss, cross_entropy_loss_der, cross_entropy_loss_deriv, softmax_crossentropy_der, mean_square_error, mean_square_error_der



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



#netter = Net_Proper()
#netter = Net_Proper2(trainer_images, trainer_labels, 10, 10, ReLU(), ReLU(),Softmax_CrossEntropy(), 1, 0.3, cross_entropy_loss, softmax_crossentropy_der)
#trainedparameters = netter.train3(shower= True, graph=False )
#print(len(trainedparameters))
#print("trainedparameters: [0]: ", trainedparameters[0].shape)
#weights1, biases1, weights2, biases2, weights3, biases3 = trainedparameters[0],trainedparameters[1],trainedparameters[2],trainedparameters[3], trainedparameters[4], trainedparameters[5]
#
#print(netter.test1(weights1, biases1,weights2, biases2, weights3, biases3, tester_images, tester_labels, True))
#netter.test4(trainedparameters, tester_images, tester_labels, True)

print("con cargador")
neuronas = [10, 10]
activaciones = [ReLU(), Tanh(), Softmax_CrossEntropy()]
netter2 = Net_Propper(trainer_images, trainer_labels, neuronas,activaciones, 500, 0.7, cross_entropy_loss, softmax_crossentropy_der)
#netter2.control_function()

traindewei, trainedbi = netter2.train(True, True)

#netter2.shower()
print(netter2.test(traindewei,trainedbi, tester_images, tester_labels, True))





    
    















