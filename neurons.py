import numpy as np



class Dense(): 
    """Neuron propper
    """
    def __init__(self, n_inputs, n_neurons, batchsize):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = 0.10 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros([n_neurons,1])
        #self.biases = np.random.randn(n_neurons, 1) 
        self.batchsize = batchsize
    def __str__(self) -> str:
        return f"Dense({self.n_inputs}, {self.n_neurons}, {self.batchsize})"
    def forward(self, inputs, testing=False ,weightstest = None, biasestest = None):
        self.stimulus = inputs
        
        if testing:
           self.potential = np.dot(weightstest , inputs) + biasestest 
        else:
            
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
        self.weights -= learning_rate * self.weight_gradient 
        self.biases -= learning_rate * (np.sum(output_gradient, axis=1, keepdims= True)* 1 / self.batchsize)
        
        return self.input_gradient
    
    