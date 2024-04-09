import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from neurons import Dense
from activations import Softmax

array1 = np.array([[1, 2],[2, 2]])
array2 = np.array
receptaculos2=np.array([[1,0,1],[0,2,0],[0,0,3],[0,-4,0],[0,0,5]]).T
retorno = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]]).T

pruebamatriz = np.array([[4,0,1],[0,2,0],[0,1,3],[5,-4,2],[3,10,5]]).T
"""print(pruebamatriz)
pruebamatrizt = pruebamatriz.T
print(pruebamatrizt)
pruebamatrizfilterde1 = np.array
print(pruebamatrizt.shape[0])
print(pruebamatrizt[1])
columnas = pruebamatrizt.shape[0]
print(type(columnas))
for x in range(columnas):
    print(x)
    filtered = np.where(pruebamatrizt[x]==pruebamatrizt[x].max(),1 , 0)
    print(pruebamatrizt[x])
    print(filtered)
    
    try:
        pruebamatrizfilterde1 = np.vstack((pruebamatrizfilterde1,filtered))
    except:
        pruebamatrizfilterde1 = filtered
print(pruebamatrizfilterde1)
    

print(pruebamatriz.T[0].max(axis=0))
pruebafiltered = np.where(pruebamatriz.max(axis=0),1,0)"""
#print(pruebafiltered)

x1 = Dense(2,2,10)
x2 = Softmax()

if isinstance(x1,Dense):
    print("yes")
else:
    print("no")
    
if isinstance(x2,Dense):
    print("yes")
else:
    print("no")
    
thislist = ["apple", "banana", "cherry"]
for x in thislist:
    counter = str(x)
    print(x)
listacounter = []
for i in range(len(thislist)):
    
    print((i))
    counter = Dense(2,2,10)
    listacounter.append(counter)
    print(thislist[i])
    print(listacounter)
    
thislist.reverse()
print(thislist)


