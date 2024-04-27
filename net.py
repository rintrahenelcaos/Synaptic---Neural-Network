import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


from neurons import Dense
from activations import mean_square_error

from config import *





def db_adapter(db):
    """Reshapes db into columns

    Args:
        db (db): keras db

    Returns:
        tuple(list): (trainer_images, trainer_labels, test_images, test_labels)
    """
    
    (train_images, train_labels), (test_images, test_labels) = db.load_data()

    trainer_labels = (np.eye(10)[train_labels]).T
    tester_labels = (np.eye(10)[test_labels]).T

    train_images = train_images/255
    test_images = test_images/255

    trainer_images = train_images.reshape(60000,784).T
    tester_images = test_images.reshape(10000, 784).T
    
    return trainer_images, trainer_labels, tester_images, tester_labels
    
def net_controller():
    """Controls consistency in NN and launches it
    """
    print("Welcome to Synapsis", end = "\n")
    print("\n")
    
    if len(NEURONS)+1 == len(ACTIVATIONS):
        print("="*50+" Training "+"="*50, end= "\n")
        print("\n")
    
        data = DATA

        trainer_images, trainer_labels, tester_images, tester_labels = db_adapter(data)
    
        netter = Net_Propper(trainer_images, trainer_labels, NEURONS, ACTIVATIONS, EPOCHS, LEARNING_RATE, LOSS_FUNC, LOSS_DER_FUNC)
        
        traindewei, trainedbi = netter.train(SHOWER, GRAPH)
        
        if TEST:
        
            netter.test(traindewei,trainedbi, tester_images, tester_labels, VS)  
    else: print("Error: Number of layers must be equal to number of assign activations minus 1")
    


class Net_Propper():
    def __init__(self, inputtrain, expected, layerneurons, activations, epochs, learning_rate, losscalc, losscalder, show_net = True):
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
                
            else:
                
                self.neurons.append(Dense(layerneurons[x-1], layerneurons[x], self.batchsize))
                
        self.neurons.append(Dense(layerneurons[-1], self.layerout, self.batchsize))
        
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
        
        if show_net: self.control_function()
        
        
    
    def control_function(self):
        """ Shows Neural Network before training
        """
        print("Database to evaluate: ", DATA.__doc__, end= "\n")
        print("Network to train:")
        for i in range(0,len(self.network), 2):
            
            print(str(self.network[i])+", activation function: ", self.network[i+1], end = "\n")
        print("\n")
        
    
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

        print("Progress: ")
        
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
            else:
                
                self.progress(int((e+1)/self.epochs*100))
                
            self.weightslist.reverse()
            self.biaslist.reverse()
        
                
        if graph:
            self.grapher(errorrs)
        print("\n")
        print("Training complete", end = "\n")
        print(f"Final error: {error}")

        return self.weightslist, self.biaslist 
    
    def progress(self, percent=0, width=200):
        """Shows progress bar

        Args:
            percent (int, optional): Previous progress. Defaults to 0.
            width (int, optional): Progress bar width. Defaults to 100.
        """
        left = width * percent // 100
        right = width - left
        
        print('\r[', '#' * left, ' ' * right, ']',
              f' {percent:.0f}%',
              sep='', end='', flush=True)
    
    def results_translator(self, output):  
        """ Converts results data in one hot matrix

        Args:
            output (np.array): results array

        Returns:
            np.array: one hot matrix
        """
        
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
        """ Compares the results of the test

        Args:
            trysmatrix (np.array): one hot matrix
            y (list): expected labels

        Returns:
            int: amount of positive matches
        """
        
        self.positives = 0
        yt = y.T
        self.trys = trysmatrix.shape[0]
        for x in range(0,10000):
            
            
            if (trysmatrix[x]==yt[x]).all():
                self.positives += 1
        return self.positives
    
    def test(self, trainedweihgtlist, trainedbiaseslist, testx, testy, vs = True):  
        """ Test model

        Args:
            trainedweihgtlist (list): list of weights
            trainedbiaseslist (list): list of biases
            testx (list): data to test
            testy (list): test labels
            vs (bool, optional): Show accuracy as a %. Defaults to True.

        Returns:
            tuple: results of test
        """
        
        
        error = 0
        

        output = testx
        i = 0
        
        for layer in self.network:
            output = layer.forward(output, True, trainedweihgtlist[i], trainedbiaseslist[i])
            
            if not(isinstance(layer, Dense)):
                
                i += 1
                
                
                    
        self.pruebaoutput= output    
        error = self.losscalc(testy, output)
        mse = mean_square_error(testy, output)
        print("\n")
        print("="*50+" Testing "+"="*50, end= "\n")
        print("\n")
        print("Test Ended", end = "\n")
        
        if vs:
            self.results_vs_y(self.results_translator(output), testy)
            
            print("Accuracy: ", np.round((self.positives/self.trys)*100,2),"%")
            print("Fails: ", 100-np.round((self.positives/self.trys)*100,2),"%")
        print ("Test Results:  error: "+ str(error)+ "\nmse: ", mse)
    
    def grapher(self, errors, initial = None, final = None):
        """ Creates graph to show error in training

        Args:
            errors (list): list of errors at training
            initial (int, optional): Initial epoch. Defaults to None.
            final (int, optional): Final epoch. Defaults to None.
        """
        
        fig, ax = plt.subplots()  
        ax.plot(errors[initial:final])  
        plt.show()
        
            
            
        



 
if __name__ == "__main__":
       
    net_controller()
        
        
        





