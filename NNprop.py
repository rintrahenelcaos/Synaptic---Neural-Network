import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt



"""entrada = [["a1","a2","a3","a4"],
           ["b1","b2","b3","b4"],
           ["c1","c2","c3","c4"],
           ["d1","d2","d3","d4"]]

entrada2=(np.array(entrada).T)

pesos = [["p11","p12","p13","p14","p15"],
         ["p11","p12","p13","p14","p15"],
         ["p11","p12","p13","p14","p15"],
         ["p11","p12","p13","p14","p15"]]

print(np.array(entrada).shape)
pesos2 = np.array(pesos)
print(pesos2.size)
print(np.dot(pesos2.T,entrada2))"""

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

class Layer():
    def __init__(self) -> None:
        self.input = None
        self.output = None
    def forward(self, input, *argv):
        pass
    def backward(self, output_gradient, learning_rate):
        pass
    def test(self):
        pass


class Dense(): #layers
    def __init__(self, n_inputs, n_neurons, batchsize):
        self.weights = 0.10 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.random.randn(n_neurons, 1)
        self.batchsize = batchsize
    def forward(self, inputs, weightstest = None, biasestest = None):
        self.stimulus = inputs
        self.potential = np.dot(self.weights , inputs) + self.biases #layer_output
        #print(self.weight @ stimulus)
        
        #self.potentialparc = np.dot(receptaculos.T , stimulus) +self.biases
        #self.potentialmanual = np.dot(receptaculos , stimulus) + bias
        return self.potential
    def test(self, inputs, weightstest, biasestest):
        self.potential = np.dot(weightstest , inputs) + biasestest
        return self.potential
    
    
    def backward(self, output_gradient, learning_rate): #returns gradient backwards -> neuron evolution
        self.weight_gradient = 1 / self.batchsize * np.dot( output_gradient , self.stimulus.T)
        self.input_gradient = np.dot( self.weights.T , output_gradient)  #input_gradient
        self.weights -= learning_rate*self.weight_gradient 
        self.biases -= learning_rate * (np.sum(output_gradient, axis=1, keepdims= True)* 1 / self.batchsize)
        
        return self.input_gradient

class Activation_Layer():

    def __init__(self, activation, activation_der):
        self.activation = activation
        self.activation_der = activation_der
        
    def forward(self, input_dense, *argv):
        self.input_dense = input_dense
        self.output_activation = self.activation(input_dense)
        return self.output_activation
        
    def backward(self, output_grad, learning_rate):
        self.activation_grad = np.multiply(output_grad , self.activation_der(self.input_dense))
        return self.activation_grad

class Tanh(Activation_Layer):
    def __init__(self):
        def tanh(x):
            tanh = np.tanh(x)
            return tanh
        def tanh_der(x):
            tanh_der = 1 - np.tanh(x) **2
            return tanh_der

        super().__init__(tanh, tanh_der)

class ReLU(Activation_Layer):
    def __init__(self):
        def relu(x):
            relufunc = np.maximum(x, 0)
            return relufunc
        def relu_der(x):
            reluder = (x > 0)*1
            return reluder
        super().__init__(relu, relu_der)

def mean_square_error(y, y_pred):
    mse = np.mean(np.power(y - y_pred, 2))
    return mse

def mean_square_error_der(y, y_pred):
    mseder = 2 * (y_pred - y) / np.size(y)
    return mseder

def cross_entropy(y, y_pred):
    xentropy = np.sum(np.log(y_pred), axis=0, keepdims=False)
    return xentropy





#pulsion = 3
#receptores=5
#receptaculos=np.array([[1,0,0,0,1],
#                      [0,2,0,0,0],
#                      [0,0,3,0,0],
#                      [0,0,0,4,0]])
#
##receptaculos=np.array([[1,0,0,0,1],[0,2,0,0,0],[0,0,3,0,0]  ])
##receptaculos=np.array([[1,0,0,0,1],[0,2,0,0,0]])
#
#

#pulso =np.array( [[1,2,3],
#          [2.0,5.0,-1.0],
#          [-1,7,3]
#          ])
#receptaculos=np.array([[1,0,1],[0,2,0],[0,0,3],[0,4,0],[0,0,5]])

#bias = np.array([[1],[2],[3],[4],[5]])
##bias = np.array([[1],[2],[3],[4]])
#
#temp_gradient = np.array([[0.1],[0.2],[0.1],[0.5],[-1]])

#temp_gradient= np.random.randn(receptores,pulsion)
#print(pulso)
#print(np.dot(pulso,receptaculos.T))
#print(receptaculos)
#print(pulso)

#print(temp_gradient*bias)
#print(np.multiply(temp_gradient,bias))

#capa1 = Dense(n_neurons=receptores,n_pulses=pulsion)
#activcapa = Tanh()
#capa2 = Dense(4,receptores)
#print(capa1.weights)
##print("weights 1",capa1.weight)
##print(capa1.biases)
#capa1.forward(pulso)
#activcapa.forward(capa1.potential)
##print(capa1.weights)
#capa2.forward(capa1.output_activation)
#error = mean_square_error(y_true, capa2.potential)
##print(error)
#error_back =mean_square_error_der(y_true, capa2.potential)
#capa2.backward(error_back, 0.5)
##print(capa2.input_gradient)
#activcapa.backward(capa2.input_gradient,0.5)
##print(activcapa.activation_grad)
#capa1.backward(activcapa.activation_grad,0.5)
#print(capa1.weights)

#network = [Dense(receptores,pulsion),Dense(4,receptores)]
#
#output = pulso
#
#for layer in network:
#    output = layer.forward(output)
#
#print(mean_square_error(y_true, output))
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

def trainbatches():
    pass

def train3(inputtrain, expected, capa1neurons, capa2neurons, activ1, activ2, epochs, learning_rate, losscalc, losscalder, shower = False):
    
    inputs = inputtrain.shape[0]
    batchsize = inputtrain.shape[1]
    capa1 = Dense(inputs, capa1neurons, batchsize)
    capa2 = Dense(capa1neurons, capa2neurons, batchsize)
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




receptaculos2=np.array([[1,0,1],[0,2,0],[0,0,3],[0,4,0],[0,0,5]])
Ypred =  np.array([[0.5, 0.5], [0, 1], [0.2, 0.8], [0.9, 0.1]]).T
#print(receptaculos2)
#receptder = (receptaculos2>0)*1
#print(receptder)
##train()
##train2()
##test()
##print(X2)
print(Ypred)
cross_entropy(Y2, Ypred)

#print(Y2.shape[1])
#print(Y2.shape[0])
#print("mean",np.mean(np.power(pulsosimp, 2)))
#print(pulso)
#print(pulso.shape)
#pulsocomp = np.sum(pulso, axis=1, keepdims= True)
#print(pulso)
#print(np.array([[1, 0]]).T)
#weights1, biases1, weights2, biases2 = train3(X2, Y2, 3, 2, Tanh(), Tanh(), 20000, 0.1, mean_square_error, mean_square_error_der, shower= True )
#print(weights1, biases1, weights2, biases2)
#print(test(weights1, biases1, weights2, biases2,np.array([[0, 0]]).T, np.array([[1, 0]]).T))
#print(test(weights1, biases1, weights2, biases2,np.array([[1, 0]]).T, np.array([[0, 1]]).T))
#print(test(weights1, biases1, weights2, biases2,np.array([[0, 1]]).T, np.array([[0, 1]]).T))
#print(test(weights1, biases1, weights2, biases2,np.array([[1, 1]]).T, np.array([[1, 0]]).T))   
#print(len(errorrs)) 
#
#fig, ax = plt.subplots()  # Create a figure containing a single axes.
#ax.plot(errorrs)  # Plot some data on the axes.
#plt.show()



    




