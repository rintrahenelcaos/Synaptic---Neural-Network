import tensorflow as tf
from tensorflow import keras
from keras import losses
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def softmax(x):
    return (np.exp(x).T / np.exp(x).sum(axis=-1)).T

logits = np.array([[1, 2, 3], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]]).T
receptaculos2=np.array([[1, 0, 1],[0, 2, 0],[0, 0, 3],[0, -4, 0],[0, 0, 5]])
retorno = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]]).T
receptaculoscorto = np.array([[1, 0., 0., 0., 0.]])
retornocorto = np.array([[10, 5, 3, 1, 4]])
receptaculoscorto = np.array([[0.8, 0.1, 0.001, 0.098, 0.001]])
retornocorto = np.array([[1, 0., 0., 0., 0.]])



class Softmax():
    def forward(self, input):
        self.input=input
        self.normalized = self.input - np.max(self.input)
        self.exponentials = np.exp(self.normalized)
        self.softmaxforward = (self.exponentials / np.sum(self.exponentials, axis=0))
        return self.softmaxforward
    def backward(self, output_grad, learning_rate):  #corregir la asignación de variables, output_gradient es la entrada desde la funcion de costo, no desde atras, en la función de calculo del gradiente va input desde forward
        size_output = np.size(self.input)
        #self.softmax_der = output_grad
        #self.softmax_der = np.dot((np.identity(size_output) - self.softmaxforward.T) * self.softmaxforward, output_grad)
        grad_matrix= np.tile(self.input, size_output)
        self.softmax_der = np.dot(grad_matrix * (np.identity(size_output) - np.transpose(grad_matrix)), output_grad)
        return self.softmax_der
    
def cross_entropy_loss(y, y_pred):
    m = np.shape(y[1])[0]
    xentropy_loss = (np.sum( - y *np.log(y_pred)))/m
    
    return xentropy_loss

cce = tf.keras.losses.CategoricalCrossentropy()
softmaxclass = Softmax()
salidaforward = softmaxclass.forward(logits)
xentrop = cross_entropy_loss(retorno, salidaforward)
ccepropio = cce(retorno.T, salidaforward.T)
salidaforwardtensor = (tf.nn.softmax(logits.T))

ccetensor = cce (retorno.T, salidaforwardtensor)
xentroptensor = cross_entropy_loss(retorno.T, salidaforwardtensor)
#salidaforwardsoftmaxpura = softmax(logits)
#ccepura = cce(retorno, salidaforwardsoftmaxpura)
#xentroppura =cross_entropy_loss(retorno, salidaforwardsoftmaxpura)
ccecorto = cce(retornocorto, receptaculoscorto)
xentropcorto = cross_entropy_loss(retornocorto.T, receptaculoscorto.T)

print(salidaforward)
print("crossentr prueba",xentrop)
print ("crosskeras: ", ccepropio)
#print(salidaforwardsoftmaxpura)
#print("crossentrop de afuera: ", ccepura)
#print("crossentrop de propia: ", xentroppura)
print(salidaforwardtensor)
print("crossentropy de keras", ccetensor)
print("crosspropia : ", xentroptensor)
#softmaxx = tf.nn.softmax([1, 0., 1])
#print(softmaxx)
print(logits)
print(receptaculoscorto)
print(retornocorto)
print("keras corto: ", ccecorto)
print("cross corto: ", xentropcorto)
print(np.shape(retorno[1])[0])


