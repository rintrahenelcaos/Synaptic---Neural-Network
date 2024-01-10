import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

array1 = np.array([[1, 2],[2, 2]])
array2 = np.array
receptaculos2=np.array([[1,0,1],[0,2,0],[0,0,3],[0,-4,0],[0,0,5]]).T
retorno = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]]).T



class Softmax(): # < ---- incluye crossentropy loss
    """Softmax without crossentropy loss"""
    def forward(self, input):
        self.input=input
        self.normalized = self.input - np.max(self.input)
        self.exponentials = np.exp(self.normalized)
        self.softmaxforward = self.exponentials / np.sum(self.exponentials, axis=0)
        return self.softmaxforward
    """def backward(self, output_grad, learning_rate):  #corregir la asignación de variables, output_gradient es la entrada desde la funcion de costo, no desde atras, en la función de calculo del gradiente va input desde forward
        size_output = np.size(self.softmaxforward)
        self.softmax_der = output_grad
        #self.softmax_der = np.dot((np.identity(size_output) - self.softmaxforward.T) * self.softmaxforward, output_grad)
        #grad_matrix= np.tile(self.softmaxforward, size_output)
        #self.softmax_der = np.dot(grad_matrix * (np.identity(size_output) - np.transpose(grad_matrix)), output_grad)
        return self.softmax_der"""
    """def backward(self, output_grad, learning_rate):
        self.softmax_der = np.array
        for i in self.softmaxforward.T:
            vertical_i = self.softmaxforward.T[i].reshape(-1,1)
            vertical_outgrad = output_grad.T[i].reshape(-1,1)
            tiled = np.tile(vertical_i, (1,len(vertical_i)))
            identityyyyy = np.identity(len(vertical_i))
            gradmatrix = tiled * (identityyyyy - tiled.T)
            gradvector = np.dot(gradmatrix, vertical_outgrad)
            try:
                self.softmax_der = np.vstack((self.softmax_der,gradvector.T))
            except:
                self.softmax_der = gradvector.T
        self.activation_grad = self.softmax_der.T
        return self.activation_grad"""
    def backward(self, output_grad, learning_rate):
        self.softmax_der = np.array
        for i in range(np.shape(self.softmaxforward[0])[0]):
            
            vertical_i = i.reshape(-1,1)
            tiled = np.tile(vertical_i, (1,len(vertical_i)))
            identityyyyy = np.identity(len(vertical_i))
            gradmatrix = tiled * (identityyyyy - tiled.T)
            gradvector = np.dot(gradmatrix, output_grad)
            try:
                self.softmax_der = np.vstack((self.softmax_der,gradvector.T))
            except:
                self.softmax_der = gradvector.T
        self.activation_grad = self.softmax_der.T
        return self.activation_grad
    
"""print(receptaculos2)
print(receptaculos2.T[1])
print(np.shape(receptaculos2[0])[0])"""
"""for i in range(np.shape(receptaculos2[0])[0]):
    print(receptaculos2.T[i].reshape(-1,1))"""

softmax = Softmax()
forwarded = softmax.forward(receptaculos2)
print(forwarded)
print(np.shape(forwarded[0])[0])
for i in range(np.shape(forwarded[0])[0]):
    vertical_i = forwarded.T[i].reshape(-1,1)
    print(vertical_i)