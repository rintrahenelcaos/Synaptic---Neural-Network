import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras




receptaculos2=np.array([[1,0,1],[0,2,0],[0,0,3],[0,-4,0],[0,0,5]]).T
retorno = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]]).T
receptaculoscorto = np.array([[1, 0, 1]]).T
retornocorto = np.array([[1, 0, 1]]).T

"""print(np.tile(receptaculoscorto, (1,2)))
print(np.tile(receptaculos2, (1,2)))
print(np.tile(receptaculos2, (1,len(receptaculoscorto))))
tiled = np.tile(receptaculos2, (1,len(receptaculoscorto)))
calculo = tiled * (np.identity(len(receptaculos2))-np.transpose(tiled))
print(calculo)"""

class Softmax():
    def forward(self, input):
        self.input=input
        self.normalized = self.input - np.max(self.input)
        self.exponentials = np.exp(self.normalized)
        self.softmaxforward = self.exponentials / np.sum(self.exponentials, axis=0)
        return self.softmaxforward
    def backward2(self, output_grad, learning_rate):  #corregir la asignación de variables, output_gradient es la entrada desde la funcion de costo, no desde atras, en la función de calculo del gradiente va input desde forward
        size_output = np.size(self.input)
        #self.softmax_der = output_grad
        #self.softmax_der = np.dot((np.identity(size_output) - self.softmaxforward.T) * self.softmaxforward, output_grad)
        grad_matrix= np.tile(self.input, size_output)
        self.softmax_der = np.dot(grad_matrix * (np.identity(size_output) - np.transpose(grad_matrix)), output_grad)
        return self.softmax_der
    def backward(self, output_grad, learning_rate):
        self.softmax_der = np.array
        for i in self.input.T:
            vertical_i = i.reshape(-1,1)
            tiled = np.tile(vertical_i, (1,len(vertical_i)))
            identityyyyy = np.identity(len(vertical_i))
            gradmatrix = tiled * (identityyyyy - tiled.T)
            gradvector = np.dot(gradmatrix, output_grad)
            try:
                salida = np.vstack((salida,gradvector.T))
            except:
                salida = gradvector.T
        return self.softmax_der
            

def prueba_crossinv(ingresa, derivativeloss):
    salida = np.array
    for i in ingresa.T:
        vertical_i = i.reshape(-1,1)
        tiled = np.tile(vertical_i, (1,len(vertical_i)))
        identityyyyy = np.identity(len(vertical_i))
        gradmatrix = tiled * (identityyyyy - tiled.T)
        gradvector = np.dot(gradmatrix, derivativeloss)
        horizontal_i = i
        print(vertical_i)
        print(tiled)
        print(gradmatrix)
        print(gradvector)
        try:
            salida = np.vstack((salida,gradvector.T))
        except:
            salida = gradvector.T
        
        print(salida)
    print(salida.T)       
    
class Softmax_CrossEntropy():
    def forward(self, input_dense):
        self.input_dense = input_dense
        self.normalized = self.input_dense - np.max(self.input_dense)
        self.exponentials = np.exp(self.normalized)
        self.output_activation = self.exponentials / np.sum(self.exponentials, axis=0)
        return self.output_activation
    def backward(self, output_grad, learning_rate):
        self.activation_grad = np.multiply(output_grad , self.input_dense)
        return self.activation_grad

def cross_entropy_loss(y, y_pred):
    m = np.shape(y[1])[0]
    xentropy_loss = (np.sum( - y *np.log(y_pred)))/m
    
    return xentropy_loss

def cross_entropy_loss_der(y, y_pred):
    xentropy_der = np.sum(y * 1/y_pred,axis=1, keepdims=True)*1/y_pred.shape[1]
    return xentropy_der

def none_softmax_crossentropy(y, y_pred):
    xentropy_loss = np.mean(np.sum( - y *np.log(y_pred)))
    return xentropy_loss

def softmax_crossentropy_der(y, y_pred):
    softmax_crossentropy_der = y_pred - y
    return softmax_crossentropy_der


    



columna = receptaculos2[:,0]
columna1 = columna.reshape(-1,1)
columnaretorno = retorno[:,0]
columnaretorno1 = columnaretorno.reshape(-1,1)
softmax = Softmax()
salidaforward = softmax.forward(columna1)
xentrop = cross_entropy_loss(columnaretorno1, salidaforward)
derxentrop = cross_entropy_loss_der(columnaretorno1, salidaforward)
final = softmax.backward(derxentrop, 0.1)

softmaxconjunta = Softmax_CrossEntropy()
xentrop2 = cross_entropy_loss(columnaretorno1, softmaxconjunta.forward(columna1))
derxentrop2 = softmax_crossentropy_der(columnaretorno1, softmaxconjunta.forward(columna1))
final2 = softmaxconjunta.backward(derxentrop2, 0.1)

print("hola")
print(receptaculos2)

print(columna)

print(columna1)
print(columna1.shape)
tiled = np.tile(columna1, (1,len(columna1)))

tiledsalidaforward =  np.tile(salidaforward, (1,len(salidaforward)))
print(tiledsalidaforward)
tiledsalidaforwardtranspose = np.transpose(tiledsalidaforward)
identityyyyy = np.identity(len(salidaforward))
print(tiledsalidaforward, " * {", identityyyyy, " - ", tiledsalidaforwardtranspose, "}")
resultado = tiledsalidaforward * (identityyyyy - tiledsalidaforwardtranspose)
print("resultado" , resultado)
print(salidaforward)
salida3 = np.dot(resultado, derxentrop)
print(salida3)
#print(final2)
prueba_crossinv(receptaculos2, retornocorto)
print(receptaculos2)
print(retornocorto)

"""
derivativesoftmaxparcial = tiledsalidaforward * (np.identity(len(salidaforward)) - np.transpose(tiledsalidaforward))
print(derivativesoftmaxparcial)
"""