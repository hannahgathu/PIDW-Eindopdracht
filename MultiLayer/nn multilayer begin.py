import autograd.numpy as np # om autograd te kunnen gebruiken
from autograd import grad
from MultiLayerNeuralNetwork import MultiLayerNeuralNetwork
from copy import deepcopy

# invoer -> 2
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0, 0, 0, 1]) # OR

# invoer -> 3
X = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
#Y = np.array([0, 1, 1, 1, 1, 1, 1, 1]) # minimaal 1
Y = np.array([0, 0, 0, 1, 0, 1, 1, 1])# minimaal 2

netwerk = MultiLayerNeuralNetwork([10, 8], X, Y, 1, 1)
#netwerk.train(100, 1e0)

 

