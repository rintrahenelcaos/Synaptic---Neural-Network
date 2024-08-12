##################################################################
############### Synapsis configuration file ######################
##################################################################


from tensorflow import keras
from keras.datasets import mnist, fashion_mnist

from activations import Activation_Layer, Softmax, Softmax_CrossEntropy, Sigmoid, Tanh,ReLU
from activations import cross_entropy_loss, cross_entropy_loss_der, softmax_crossentropy_der, mean_square_error, mean_square_error_der

##################### Config bellow ##############################

DATA = mnist
"""Db to train and test. Options:  
                                mnist
                                fashion_mnist"""

EPOCHS = 200
"""Training epochs"""

LEARNING_RATE = 0.5
"""learning rate"""

NEURONS = [10, 5]
"""Neurons of each layer. Last layer is assign automatically"""

ACTIVATIONS = [ReLU(), Tanh(), Softmax_CrossEntropy()]
"""Activations functions for each layer. Options: Softmax()
                                                  Softmax_CrossEntropy() 
                                                  Sigmoid()
                                                  Tanh()
                                                  ReLU()
    Softmax_CrossEntropy is used in conjunction with softmax_crossentropy_der as LOSS_DER_FUNC"""

LOSS_FUNC = cross_entropy_loss
"""Loss function. Options: cross_entropy_loss
                           mean_square_error"""

LOSS_DER_FUNC = softmax_crossentropy_der
"""Loss function derivative: cross_entropy_loss_der
                             softmax_crossentropy_der
                             mean_square_error_der
   softmax_crossentropy_der is meant to be used only with Softmax_CrossEntropy as last activation function in the network"""

SHOW_NET = True
"""Show Neural Network to train in console."""

SHOWER = False
"""Show each epoch with its error instead of progress bar"""

GRAPH = False
"""Show error graph after training"""

TEST = True
"""Test after training"""

VS = True
"""Show accuracy results of test"""

