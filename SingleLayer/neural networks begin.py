import numpy as np
from NeuralNetwork import NeuralNetwork

# invoer voor OF operatie (traininginput)
X = np.array([  [0,0],
                [0,1],
                [1,0],
                [1,1]])
    
# uitvoer OF (trainingoutput)       
Y = np.array([0,1,1,1])
    
n = NeuralNetwork(X, Y)
n.train(1000, 1e1)
print(n.predict([1,1]))
print('Weights               : ', n.p)
print('Output training data  : ', Y)
print('Output after training : ', 1/(1+np.exp(-(np.dot(X,n.p[:-1]) + n.p[-1]))))



<<<<<<< HEAD:neural networks begin.py
### invoer voor EN operatie (traininginput)
##X = np.array([  [0,0],
##                [0,1],
##                [1,0],
##                [1,1]])
##    
### uitvoer EN (trainingoutput)       
##y = np.array([0,0,0,1])
=======
# invoer voor EN operatie (traininginput)
X = np.array([  [0,0],
                [0,1],
                [1,0],
                [1,1]])
    
# uitvoer EN (trainingoutput)       
Y = np.array([0,0,0,1])
>>>>>>> origin/master:SingleLayer/neural networks begin.py
    
n = NeuralNetwork(X, Y)
n.train(1000, 1e1)
print(n.predict([1,1]))
print('Weights               : ', n.p)
print('Output training data  : ', Y)
print('Output after training : ', 1/(1+np.exp(-(np.dot(X,n.p[:-1]) + n.p[-1]))))
   
