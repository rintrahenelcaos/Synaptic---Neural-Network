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



class Net_Propper():
    def __init__(self, inputtrain, expected, layerneurons, activations, epochs, learning_rate, losscalc, losscalder):
        """Neural Natework generating class
        IMPORTANT: All arrays must be in columns

        Args:
            inputtrain (np.array): Training data array 
            expected (np.array): Labels data array ---> onehot
            layerneurons (list): list with the number of neurons for each layer not including the last one  -----> integer
            activations (list): list of activations functions  ---> activations module
            epochs (int): number of epochs
            learning_rate (float): adjustment between epochs
            losscalc (function): loss calculation function  -----> activations module
            losscalder (function):loss calculation function derivative for backpropagation -----> activations module
        """
        
        
        
        self.inputtrain = inputtrain
        self.expected = expected
        self.inputs = inputtrain.shape[0]
        self.batchsize = inputtrain.shape[1]
        self.layerout = expected.shape[0]
        self.numberoflayers = len(activations)
        self.neurons = []
        for x in range(len(layerneurons)):
            if x==0:
                
                self.neurons.append(Dense(self.inputs, layerneurons[x],self.batchsize))
                #print(Dense(self.inputs, layerneurons[x],self.batchsize))
            else:
                
                self.neurons.append(Dense(layerneurons[x-1], layerneurons[x], self.batchsize))
                #print(Dense(layerneurons[x-1], layerneurons[x], self.batchsize))
        self.neurons.append(Dense(layerneurons[-1], self.layerout, self.batchsize))
        #print(Dense(layerneurons[-1], self.layerout, self.batchsize))
        self.network = []
        for i in range(self.numberoflayers):
            self.network.append(self.neurons[i])
            self.network.append(activations[i])
        
        self.losscalc = losscalc
        self.losscalder = losscalder
        self.epochs = epochs
        self.learningrate = learning_rate
        self.epochcounter = 0 # not working
        self.lapmarc = 0 # not working
    
    def control_function(self):
        print(self.network)
    
    def starttrain(self, shower = False, graph = False):  # not working
        self.epochcounter = 0 
        self.lapmarc = 0
        
        self.weightslist, self.biaslist = self.train(shower, graph)
        
        return self.weightslist, self.biaslist
        
    
    def train(self, shower = False, graph = False):
        """Network Training Function

        Args:
            shower (bool, optional): Show the processing epoch/total epochs. Defaults to False.
            graph (bool, optional): Show graph of loss. Defaults to False.

        Returns:
            _type_: tuple(weightslist, biaslist) for testing
        """
        
        
        error = 0
        errorrs = []
        self.weightslist = []
        self.biaslist = []
        self.epochcounter = 0

        for e in range(self.epochs):
            self.weightslist = []
            self.biaslist = []
            self.lapmarc = 0

            output = self.inputtrain
            for layer in self.network:
                output = layer.forward(output)
                self.lapmarc+=1
            
            

            error = self.losscalc(self.expected, output)

            grad = self.losscalder(self.expected, output)

            for layer in reversed(self.network):
                grad = layer.backward(grad, self.learningrate)
                self.lapmarc+=1
                if isinstance(layer,Dense):
                    self.weightslist.append(layer.weights)
                    self.biaslist.append(layer.biases)

            error /=len(self.inputtrain)
            errorrs.append(error)
            self.epochcounter =((self.epochcounter+1)/self.epochs)*100
            if shower: 
                print(f"{e + 1}/{self.epochs}, error={error}")
            self.weightslist.reverse()
            self.biaslist.reverse()
        
                
        if graph:
            self.grapher(errorrs)
        print("training ended")

        return self.weightslist, self.biaslist 
    
    def countinlaps(self):
        
        pass
    
    def results_translator(self, output):  #converts results data in one hot matrix
        self.trysmatrixfiltered = np.array
        self.result = output.T
        trys = self.result.shape[0]
        for x in range(trys):
            filtered = np.where(self.result[x]==self.result[x].max(),1 , 0)
            try:
                self.trysmatrixfiltered = np.vstack((self.trysmatrixfiltered,filtered))
            except:
                self.trysmatrixfiltered = filtered
        return self.trysmatrixfiltered
    
    def results_vs_y(self, trysmatrix, y):  # compares the results of the test
        self.positives = 0
        yt = y.T
        self.trys = trysmatrix.shape[0]
        for x in range(0,10000):
            
            
            if (trysmatrix[x]==yt[x]).all():
                self.positives += 1
        return self.positives
    
    def test(self, trainedwiehgtlist, trainedbiaseslist, testx, testy, vs = False):  #test function for the model
        error = 0
        

        output = testx
        i = 0
        
        for layer in self.network:
            output = layer.forward(output, True, trainedwiehgtlist[i], trainedbiaseslist[i])
            
            if not(isinstance(layer, Dense)):
                
                i += 1
                
                
                    
        self.pruebaoutput= output    
        error = self.losscalc(testy, output)
        mse = mean_square_error(testy, output)
        if vs:
            self.results_vs_y(self.results_translator(output), testy)
            print("eficiencia: ", np.round((self.positives/self.trys)*100,2),"%")
            print("fallos: ", 100-np.round((self.positives/self.trys)*100,2),"%")
        return "Resultado del test:  error: "+ str(error), "//// mse: ", mse
    
    def grapher(self, errors, initial = None, final = None):
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(errors[initial:final])  # Plot some data on the axes.
        plt.show()
        
            
            
        


class Net_Proper2():
    def __init__(self, inputtrain, expected, capa1neurons, capa2neurons,activ1, activ2, activ3,epochs, learning_rate, losscalc, losscalder):
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
        self.batchsize = inputtrain.shape[1]
        self.capaout = expected.shape[0]
        self.capa1neurons = capa1neurons
        self.capa2neurons = capa2neurons
        self.acitvation1 = activ1
        self.activation2 = activ2
        self.activation3 = activ3
        
        self.capa1 = Dense(self.inputs, self.capa1neurons, self.batchsize)
        self.capa2 = Dense(self.capa1neurons, self.capa2neurons, self.batchsize)
        self.capa3 = Dense(self.capa2neurons, self.capaout, self.batchsize)
        self.network = [self.capa1, self.acitvation1, self.capa2, self.activation2, self.capa3, self.activation3]
        self.losscalc = losscalc
        self.losscalder = losscalder
        self.epochs = epochs
        self.learningrate = learning_rate
        
    def train3(self, shower = False, graph = False):
        
        
        error = 0
        errorrs = []

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

        return self.capa1.weights, self.capa1.biases , self.capa2.weights, self.capa2.biases, self.capa3.weights, self.capa3.biases
    
    def results_translator(self, output):
        self.trysmatrixfiltered = np.array
        self.result = output.T
        trys = self.result.shape[0]
        for x in range(trys):
            filtered = np.where(self.result[x]==self.result[x].max(),1 , 0)
            try:
                self.trysmatrixfiltered = np.vstack((self.trysmatrixfiltered,filtered))
            except:
                self.trysmatrixfiltered = filtered
        return self.trysmatrixfiltered
    
    def results_vs_y(self, trysmatrix, y):
        self.positives = 0
        yt = y.T
        self.trys = trysmatrix.shape[0]
        for x in range(0,10000):
            
            
            if (trysmatrix[x]==yt[x]).all():
                self.positives += 1
        return self.positives
    
    def test4(self, paramlist, testx, testy, vs = False):
        error = 0
        

        output = testx
        i = 0
        counter = 0
        for layer in self.network:
            output = layer.forward(output, True, paramlist[i], paramlist[i+1])
            print(i)
            if isinstance(layer, Dense):
                counter += 1
                print("pasos por dense: ",counter)
                if i != len(paramlist)-2:
                    i = i+2
                    
            
        error = self.losscalc(testy, output)
        if vs:
            self.results_vs_y(self.results_translator(output), testy)
            print("eficiencia: ", np.round((self.positives/self.trys)*100,2),"%")
            print("fallos: ", 100-np.round((self.positives/self.trys)*100,2),"%")
    
    def test2(self, weightstrained1, biasestrained1, weightstrained2, biasestrained2, testx, testy, vs = False):
        error = 0
        for e in range(self.epochs):

            output = self.inputtrain
            for layer in self.network:
                output = layer.forward(output)
            error = self.losscalc(self.expected, output)
        if vs:
            self.results_vs_y(self.results_translator(output), testy)
            print("eficiencia: ", np.round((self.positives/self.trys)*100,2),"%")
            print("fallos: ", 100-np.round((self.positives/self.trys)*100,2),"%")
            
    
            
        
    
    def test1(self, weights1, biases1, weights2, biases2, testx, testy, vs = False):
        error = 0
        self.inputs = testx.shape[0]
        self.capaout = testy.shape[0]
        self.batchsize = testx.shape[1]
        capa1 = Dense(self.inputs, self.capa1neurons, self.batchsize)
        activ1 = Tanh()
        capa2 = Dense(self.capa1neurons, self.capaout, self.batchsize)
        activ2 = Tanh()
        output = testx
        capa1.forward(output,True, weights1, biases1)
        self.acitvation1.forward(capa1.potential)
        capa2.forward(self.acitvation1.output_activation,True, weights2, biases2)
        output = self.activation2.forward(capa2.potential)
        self.outputtest = output
        error = self.losscalc(testy, output)
        mse = mean_square_error(testy, output)
        if vs:
            self.results_vs_y(self.results_translator(output), testy)
            print("eficiencia: ", np.round((self.positives/self.trys)*100,2),"%")
            print("fallos: ", 100-np.round((self.positives/self.trys)*100,2),"%")
        #print("Resultado del test: "+str(testy)+" vs "+str(np.round(activ2.output_activation, 3)))
        """errorizado = np.sum(abs((testy)-(np.round(activ2.output_activation, 3))))
        return ("Resultado del test: " + str((testy)-(np.round(activ2.output_activation, 3))) + "  error: "+ str(errorizado))"""
        return ("Resultado del test:  error: "+ str(error), "//// mse: ", mse)
        
    
    def test (self, weights1, biases1, weights2, biases2, testx, testy):
        error = 0
        self.inputs = testx.shape[0]
        self.capaout = testy.shape[0]
        self.batchsize = testx.shape[1]
        
        
        
        self.capa1 = Dense(self.inputs, self.capa1neurons, self.batchsize)
        self.capa2 = Dense(self.capa1neurons, self.capaout, self.batchsize) 
        self.network = [self.capa1, self.acitvation1, self.capa2, self.activation2]
        
        self.capa1.forward(testx, weights1,biases1)
        self.acitvation1.forward(self.capa1.potential)
        self.capa2.forward(self.acitvation1.output_activation, weights2, biases2)
        output = self.activation2.forward(self.capa2.potential)
        error = self.losscalc(testy, output)
        mse = mean_square_error(testy, output)
        
        
        #errorizado = np.sum(abs((testy)-(np.round(self.activation2.output_activation, 3))))
        return ("Resultado del test:  error: "+ str(error), "//// mse: ", mse)
    
    def grapher(self, errors, initial = None, final = None):
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(errors[initial:final])  # Plot some data on the axes.
        plt.show()
    



    
        
        
        





