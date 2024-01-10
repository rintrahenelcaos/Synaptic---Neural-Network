import numpy as np



class Activation_Layer():
    """Neuron layer propper
    """

    def __init__(self, activation, activation_der):
        self.activation = activation
        self.activation_der = activation_der
        
    def forward(self, input_dense):
        self.input_dense = input_dense
        self.output_activation = self.activation(input_dense)
        return self.output_activation
        
    def backward(self, output_grad, learning_rate):
        self.activation_grad = np.multiply(output_grad , self.activation_der(self.input_dense))
        return self.activation_grad


class Tanh():
    def forward(self, input_dense):
        self.input_dense = input_dense
        self.output_activation = np.tanh(input_dense)
        return self.output_activation
    def backward(self, output_grad, learning_rate):
        tanh_der = 1 - np.tanh(self.input_dense) ** 2
        self.activation_grad = np.multiply(output_grad , tanh_der)
        return self.activation_grad


class ReLU():
    def forward(self, input_dense):
        self.input_dense = input_dense
        self.output_activation = np.maximum(self.input_dense, 0)
        return self.output_activation
    def backward(self, output_grad, learning_rate):
        reluder = 1*(self.input_dense > 0)
        self.activation_grad = np.multiply(output_grad , reluder)
        return self.activation_grad


class Sigmoid():
    def forward(self, input_dense):
        self.input_dense = input_dense
        self.output_activation = 1 / (1 + np.exp(- self.input_dense))
        return self.output_activation
    def backward(self, output_grad, learning_rate):
        sigmoidf = 1 / (1 + np.exp(- self.input_dense))
        sigmoidder = sigmoidf * (1 - sigmoidf)
        self.activation_grad = np.multiply(output_grad ,sigmoidder)
        return self.activation_grad
    
class Softmax_CrossEntropy(): # < ---- softmax con crossentropy loss, usar con softmax_crossentropy_der en derivada de loss
    """Softmax with crossentropy loss, use in conjunction with softmax_crossentropy_der as loss derivative
    """
    def forward(self, input_dense):
        self.input_dense = input_dense
        self.normalized = self.input_dense - np.max(self.input_dense)
        self.exponentials = np.exp(self.normalized)
        self.output_activation = self.exponentials / np.sum(self.exponentials, axis=0)
        return self.output_activation
    def backward(self, output_grad, learning_rate):
        self.activation_grad = output_grad
        return self.activation_grad


class Softmax():   # <--- solo usar sin crossentropy loss
    """Softmax indenpendient of crossentropy loss, if crossentropy loss is requiered use cross_entropy_loss_der as derivative of loss function.
    """
    def forward(self, input):
        self.input=input
        self.normalized = self.input - np.max(self.input)
        self.exponentials = np.exp(self.normalized)
        self.softmaxforward = self.exponentials / np.sum(self.exponentials, axis=0)
        return self.softmaxforward
    
    def backward(self, output_grad, learning_rate):
        self.softmax_der = np.array
        for i in range(np.shape(self.softmaxforward[0])[0]):
            
            vertical_i = self.softmaxforward.T[i].reshape(-1,1)
            tiled = np.tile(vertical_i, (1,len(vertical_i)))
            identityyyyy = np.identity(len(vertical_i))
            gradmatrix = tiled * (identityyyyy - tiled.T)
            outvector = output_grad.T[i].reshape(-1,1)
            gradvector = np.dot(gradmatrix, outvector)
            try:
                self.softmax_der = np.vstack((self.softmax_der,gradvector.T))
            except:
                self.softmax_der = gradvector.T
        self.activation_grad = self.softmax_der.T
        return self.activation_grad


########## LOSS CALCULATORS  #############

def mean_square_error(y, y_pred):
    mse = np.mean(np.power(y - y_pred, 2))
    return mse

def mean_square_error_der(y, y_pred):
    mseder = 2 * (y_pred - y) / np.size(y)
    return mseder

def none_der(y, ypred):pass


    
#class Softmax_CrossEntropy():
#    def forward(self, input_dense):
#        self.input_dense = input_dense
#        exponentials = np.exp(self.input_dense)
#        self.output_activation = exponentials / np.sum(input)
#        return self.output_activation
#    def forward_crossentropy(self, y_true, input):
#        self.xentropy_loss = np.mean(np.sum( - y_true *np.log(self.forward(input))))
#        return self.xentropy_loss
#    
#    
#    def backward(self, output_grad, learning_rate):
#        softmax_crossentropy_der = ypred - y_true
#        return softmax_crossentropy_der
    
def cross_entropy_loss(y, y_pred):
    m = np.shape(y[1])[0]
    xentropy_loss = (np.sum( - y *np.log(y_pred)))/m
    return xentropy_loss

def softmax_crossentropy_der(y, y_pred):
    softmax_crossentropy_der = y_pred - y
    return softmax_crossentropy_der

def cross_entropy_loss(y, y_pred):
    m = np.shape(y[1])[0]
    xentropy_loss = (np.sum( - y *np.log(y_pred)))/m
    return xentropy_loss

def cross_entropy_loss_der(y, y_pred):
    m = np.shape(y[1])[0]
    xentropy_der = (-(y * 1/y_pred))/m
    return xentropy_der

def cross_entropy_loss_deriv (y, y_pred):
    xentropy_der = np.sum(-y * 1/y_pred, axis=1, keepdims=True)/y_pred.shape[1]
    return xentropy_der

def softmaxuda(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


##### Pruebas #####

"""receptaculos2=np.array([[1,0,1],[0,2,0],[0,0,3],[0,-4,0],[0,0,5]]).T
retorno = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]]).T
receptaculoscorto = np.array([[2, 1, 1]]).T
retornocorto = np.array([[1, 2, 1]]).T

softmax = Softmax()
xentrop = cross_entropy_loss(retorno, softmax.forward(receptaculos2))
derxentrop = cross_entropy_loss_der(retorno, softmax.forward(receptaculos2))
final = softmax.backward(derxentrop, 0.1)

softmaxconjunta = Softmax_CrossEntropy()
xentrop2 = cross_entropy_loss(retorno, softmaxconjunta.forward(receptaculos2))
derxentrop2 = softmax_crossentropy_der(retorno, softmaxconjunta.forward(receptaculos2))
final2 = softmaxconjunta.backward(derxentrop2, 0.1)

sigmoide = Sigmoid()"""

"""

#print(receptaculoscorto)
print("softmax separado")
print(softmax.forward(receptaculos2))
print("cross entropy: ",xentrop)
print("der_crossentropy",derxentrop)
print("salida")
print(final)
#print(receptaculos2*retorno)
#print(np.diagflat(receptaculos2.size))
#print(final)

print("softmax_conjunta")
print(softmaxconjunta.forward(receptaculos2))
print("crossentropy",xentrop2)
print("dercrossentropy", derxentrop2)
print("salida2")
print(final2)

print("udasol: ", softmaxuda(receptaculos2))
#
#print("mse", mean_square_error_der(retornocorto, receptaculoscorto))
#print(receptaculos2.shape[1])
#print(retornocorto * 1/softmax.forward(receptaculoscorto))
#print((np.sum(retornocorto * 1/softmax.forward(receptaculoscorto),axis=1, keepdims=True))*1/receptaculoscorto.shape[1])
#print(np.sum(receptaculos2, axis=1, keepdims=True))

print("sigmoide")
print(sigmoide.forward(receptaculos2))
print("mse: ",mean_square_error(retorno,sigmoide.forward(receptaculos2)))
print("mse der: ",mean_square_error_der(retorno,sigmoide.forward(receptaculos2)))
print("salidad sigmoide")
print(sigmoide.backward(mean_square_error_der(retorno,sigmoide.forward(receptaculos2)), 0.1))

print("pruebas:")

print("dercross sobre retorno corto: ", cross_entropy_loss_der(retornocorto, receptaculoscorto))
dercross = cross_entropy_loss_der(retornocorto, receptaculoscorto)
print(softmax.forward(receptaculoscorto))
print (softmax.backward(retornocorto, 0.1))
print(np.size(retornocorto))
print(len(retornocorto))
print(np.tile(retornocorto, np.size(retornocorto)))
print (retornocorto * (np.identity(np.size(retornocorto)) - np.transpose(retornocorto)))


"""