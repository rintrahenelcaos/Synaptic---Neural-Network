from net import Net_Proper, Net_Proper2
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


X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
Y2 = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]).T
batch = tuple (zip(X, Y))




#netter = Net_Proper()
netter = Net_Proper2(trainer_images, trainer_labels, 10, ReLU(), Softmax(), 5, 0.5, cross_entropy_loss, cross_entropy_loss_der)

weights1, biases1, weights2, biases2 = netter.train3(shower= True, graph=True )
"""
(train_images, train_labels), (test_images, test_labels) = data.load_data()
#eyematrix = (np.eye(10)[train_labels]).T
#print(type(eyematrix))
#print(eyematrix.shape)
#print(eyematrix[:,:5])
#print(train_images.shape)
#print(train_images[1])

trainer_labels = (np.eye(10)[train_labels]).T
tester_labels = (np.eye(10)[test_labels]).T

train_images = train_images/255
test_images = test_images/255

pruebareshape = train_images[0]

#print(pruebareshape.shape)

pruebareshape = pruebareshape.reshape(784,)
#print(pruebareshape.shape)
#print(train_images[0])
#print(pruebareshape)

trainer_images = train_images.reshape(60000,784).T
trainer_imagesT = trainer_images.T
print(tester_images.shape)
print(X2.shape)
#print("reshape",train_images.shape," to ",trainer_images.shape)
#print(trainer_images[0].shape)


"""

