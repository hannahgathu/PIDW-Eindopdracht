import autograd.numpy as np
from autograd import grad
from TwoLayerNeuralNetwork import TwoLayerNeuralNetwork
from NeuralNetwork import NeuralNetwork
from read_ubyte import *

Y = Y/9
X = X/255

n = TwoLayerNeuralNetwork(5, X, Y, 1, 1)
n.train(1000, 0.01)
uit = np.zeros(np.shape(X)[0])
verschil = np.zeros(np.shape(X)[0])
for i in range(np.shape(X)[0]):
    uit[i] = n.predict(X[i])
    verschil[i] = round(9 * uit[i]) - round(9 * Y[i])
print(np.count_nonzero(verschil))
