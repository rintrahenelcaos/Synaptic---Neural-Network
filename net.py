import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt

from neurons import Dense
from activations import Tanh
from activations import mean_square_error, mean_square_error_der, cross_entropy_loss, softmax_crossentropy_der, cross_entropy_loss_der
from activations import ReLU
from activations import Sigmoid
from activations import Softmax_CrossEntropy
from activations import Softmax


pulsosimp =np.array( [[1,2,3]]).T
pulso =np.array( [[1,2,3,4,5],
          [2.0,5.0,-1.0,2.0,0],
          [-1,7,3,-0.8,1]
          ]).T

y_true = np.array([[1,0,0],
                   [0,1,0],
                   [0,0,1]]) . T

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
Y2 = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]).T
batch = tuple (zip(X, Y))
errorrs = []

def train(inputs, expected, capa1neurons, capa2neurons,activ1, activ2, inputtrain, epochs, learning_rate, losscalc, losscalder, shower = False):
    
    batchsize = inputtrain.shape[1]
    capa1 = Dense(inputs, capa1neurons, batchsize)
    capa2 = Dense(capa1neurons, capa2neurons, batchsize)
    network = [capa1, activ1, capa2, activ2]

    error = 0
    
    for e in range(epochs):
        
        for x, y in zip(inputtrain, expected):
            #error = 0
            output = x
            for layer in network:
                output = layer.forward(output)

            error = losscalc(y, output)

            grad = losscalder(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(inputtrain)
        errorrs.append(error)
        if shower:
            print(f"{e + 1}/{epochs}, error={error}")
    
    return capa1.weights, capa1.biases , capa2.weights, capa2.biases



def train2():
    capa1 = Dense(2,3)
    activ1 = Tanh()
    capa2 = Dense(3, 1)
    activ2 = Tanh()

    epochs = 1000
    learning_rate = 0.2
    

    for e in range(epochs):
        error = 0
        
        
        for x, y in zip(X, Y):
            output = x
            rescapa1=capa1.forward(x)
            resactiv1 = activ1.forward(rescapa1)
            rescapa2 = capa2.forward(resactiv1)
            output = activ2.forward(rescapa2)
        
            error = mean_square_error(y, output)

            grad = mean_square_error_der(y, output)
        
            
            gradactiv2 = activ2.backward(grad, learning_rate)
            gradcapa2 = capa2.backward(gradactiv2, learning_rate)
            gradactiv1 = activ1.backward(gradcapa2, learning_rate)
            capa1.backward(gradactiv1, learning_rate)

        error /= len(x)
        errorrs.append(error)
        print(f"{e + 1}/{epochs}, error={error}")
             
    return capa1.weights, capa1.biases , capa2.weights, capa2.biases


def train3(inputtrain, expected, capa1neurons, activ1, activ2, epochs, learning_rate, losscalc, losscalder, shower = False):
    
    inputs = inputtrain.shape[0]
    batchsize = inputtrain.shape[1]
    capaout = expected.shape[0]
    capa1 = Dense(inputs, capa1neurons, batchsize)
    capa2 = Dense(capa1neurons, capaout, batchsize)
    network = [capa1, activ1, capa2, activ2]

    error = 0
    
    for e in range(epochs):

        output = inputtrain
        for layer in network:
            output = layer.forward(output)
        
        error = losscalc(expected, output)

        grad = losscalder(expected, output)

        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

        error /=len(inputtrain)
        errorrs.append(error)
        if shower: 
            print(f"{e + 1}/{epochs}, error={error}")
    
    return capa1.weights, capa1.biases , capa2.weights, capa2.biases
        


def test(weights1, biases1, weights2, biases2, testx, testy):
    batchsize = testy.shape[1]
    capa1 = Dense(2, 3, batchsize)
    activ1 = Tanh()
    capa2 = Dense(3, 1, batchsize)
    activ2 = Tanh()
    output = testx
    capa1.test(output, weights1, biases1)
    activ1.forward(capa1.potential)
    capa2.test(activ1.output_activation, weights2, biases2)
    activ2.forward(capa2.potential)
    #print("Resultado del test: "+str(testy)+" vs "+str(np.round(activ2.output_activation, 3)))
    errorizado = np.sum(abs((testy)-(np.round(activ2.output_activation, 3))))
    return ("Resultado del test: " + str((testy)-(np.round(activ2.output_activation, 3))) + "  error: "+ str(errorizado))

class Net_Proper2():
    def __init__(self, inputtrain, expected, capa1neurons, activ1, activ2, epochs, learning_rate, losscalc, losscalder):
        """ IMPORTANT: All arrays must be in columns
        inputtrain: Training data array 
        expected: Labels data array ----> onehot
        capa1neurons: number of neurons  -----> integer
        activ1: Activation for first layer  ------> activations module 
        activ2: Activation for second layer  -----> activations module
        epochs: epochs ---> integer
        learning_rate: -----> float
        losscalc: loss calculation function  -----> activations module
        losscalder: loss calculation function derivative for backpropagation -----> activations module
        """
        
        self.inputtrain = inputtrain
        self.expected = expected
        self.inputs = inputtrain.shape[0]
        batchsize = inputtrain.shape[1]
        capaout = expected.shape[0]
        self.capa1 = Dense(self.inputs, capa1neurons, batchsize)
        self.capa2 = Dense(capa1neurons, capaout, batchsize)
        self.network = [self.capa1, activ1, self.capa2, activ2]
        self.losscalc = losscalc
        self.losscalder = losscalder
        self.epochs = epochs
        self.learningrate = learning_rate
        
    def train3(self, shower = False, graph = False):
    
                

        error = 0

        for e in range(self.epochs):

            output = self.inputtrain
            for layer in self.network:
                output = layer.forward(output)

            error = self.losscalc(self.expected, output)

            grad = self.losscalder(self.expected, output)

            for layer in reversed(self.network):
                grad = layer.backward(grad, self.learningrate)

            error /=len(self.inputtrain)
            errorrs.append(error)
            if shower: 
                print(f"{e + 1}/{self.epochs}, error={error}")
                
        if graph:
            self.grapher(errorrs)

        return self.capa1.weights, self.capa1.biases , self.capa2.weights, self.capa2.biases
    
    def test(self, weights1, biases1, weights2, biases2, testx, testy):
        batchsize = testy.shape[1]
        capa1 = Dense(2, 3, batchsize)
        activ1 = Tanh()
        capa2 = Dense(3, 1, batchsize)
        activ2 = Tanh()
        output = testx
        capa1.test(output, weights1, biases1)
        activ1.forward(capa1.potential)
        capa2.test(activ1.output_activation, weights2, biases2)
        activ2.forward(capa2.potential)
        #print("Resultado del test: "+str(testy)+" vs "+str(np.round(activ2.output_activation, 3)))
        errorizado = np.sum(abs((testy)-(np.round(activ2.output_activation, 3))))
        return ("Resultado del test: " + str((testy)-(np.round(activ2.output_activation, 3))) + "  error: "+ str(errorizado))
    
    def grapher(self, errors, initial = None, final = None):
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(errors[initial:final])  # Plot some data on the axes.
        plt.show()
        


class Net_Proper():
    
    
    def train3(self, inputtrain, expected, capa1neurons, activ1, activ2, epochs, learning_rate, losscalc, losscalder, shower = False, graph = False):
    
        inputs = inputtrain.shape[0]
        batchsize = inputtrain.shape[1]
        capaout = expected.shape[0]
        self.capa1 = Dense(inputs, capa1neurons, batchsize)
        self.capa2 = Dense(capa1neurons, capaout, batchsize)
        network = [self.capa1, activ1, self.capa2, activ2]

        error = 0

        for e in range(epochs):

            output = inputtrain
            for layer in network:
                output = layer.forward(output)

            error = losscalc(expected, output)

            grad = losscalder(expected, output)

            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

            error /=len(inputtrain)
            errorrs.append(error)
            if shower: 
                print(f"{e + 1}/{epochs}, error={error}")
                
        if graph:
            self.grapher(errorrs)

        return self.capa1.weights, self.capa1.biases , self.capa2.weights, self.capa2.biases
    
    def test(self, weights1, biases1, weights2, biases2, testx, testy):
        batchsize = testy.shape[1]
        capa1 = Dense(2, 3, batchsize)
        activ1 = Tanh()
        capa2 = Dense(3, 1, batchsize)
        activ2 = Tanh()
        output = testx
        capa1.test(output, weights1, biases1)
        activ1.forward(capa1.potential)
        capa2.test(activ1.output_activation, weights2, biases2)
        activ2.forward(capa2.potential)
        #print("Resultado del test: "+str(testy)+" vs "+str(np.round(activ2.output_activation, 3)))
        errorizado = np.sum(abs((testy)-(np.round(activ2.output_activation, 3))))
        return ("Resultado del test: " + str((testy)-(np.round(activ2.output_activation, 3))) + "  error: "+ str(errorizado))
    
    def grapher(self, errors, initial = None, final = None):
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(errors[initial:final])  # Plot some data on the axes.
        plt.show()
        


relus = ReLU()
receptaculos2=np.array([[1,0,1],[0,2,0],[0,0,3],[0,-4,0],[0,0,5]]).T
retorno = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]]).T
#print(receptaculos2)
#print(receptaculos2.shape[0])
#relussalida=relus.forward(input_dense=receptaculos2)
#print(relussalida)
#print(retorno)
#reluback =relus.backward(retorno ,1)
#print( "back", reluback)

#receptder = ((reluback)>0)

#print(receptder)
#train()
#train2()
#test()
#print(X2)
#print(Y2)
#print(Y2.shape[1])
#print(Y2.shape[0])
#print("mean",np.mean(np.power(pulsosimp, 2)))
#print(pulso)
#print(pulso.shape)
#pulsocomp = np.sum(pulso, axis=1, keepdims= True)
#print(pulso)
#print(np.array([[1, 0]]).T)
#weights1, biases1, weights2, biases2 = train3(X2, Y2, 3, ReLU(), Softmax(), 20000, 1.5, cross_entropy_loss, cross_entropy_loss_der, shower= True )
#print(weights1, biases1, weights2, biases2)
#print(test(weights1, biases1, weights2, biases2,np.array([[0, 0]]).T, np.array([[1, 0]]).T))
#print(test(weights1, biases1, weights2, biases2,np.array([[1, 0]]).T, np.array([[0, 1]]).T))
#print(test(weights1, biases1, weights2, biases2,np.array([[0, 1]]).T, np.array([[0, 1]]).T))
#print(test(weights1, biases1, weights2, biases2,np.array([[1, 1]]).T, np.array([[1, 0]]).T))   
#print(len(errorrs)) 
#
#fig, ax = plt.subplots()  # Create a figure containing a single axes.
#ax.plot(errorrs[10000:])  # Plot some data on the axes.
#plt.show()



