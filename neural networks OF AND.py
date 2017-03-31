import numpy as np
from NeuralNetwork import NeuralNetwork

# invoer voor OF operatie (traininginput)
X = np.array([  [0,0],
                [0,1],
                [1,0],
                [1,1]])
    
# uitvoer OF (trainingoutput)       
y = np.array([0,1,1,1])
    
n = NeuralNetwork([1,1,1], 100, 1e0)
n.train(X, y)
print(n.predict([1,1]))
print('Weights               : ', n.p)
print('Output training data  : ', y)
print('Output after training : ', 1/(1+np.exp(-(np.dot(X,n.p[:-1]) + n.p[-1]))))



##### invoer voor EN operatie (traininginput)
####X = np.array([  [0,0],
####                [0,1],
####                [1,0],
####                [1,1]])
####    
##### uitvoer EN (trainingoutput)       
####y = np.array([0,0,0,1])
##    
##n = NeuralNetwork([1,1,1], 0, 1e1)
##n.train(X, y)
##print(n.predict([1,1]))
##print('Weights               : ', n.p)
##print('Output training data  : ', y)
##print('Output after training : ', 1/(1+np.exp(-(np.dot(X,n.p[:-1]) + n.p[-1]))))
##   
