import autograd.numpy as np
from autograd import grad
from MultiLayerNeuralNetwork import MultiLayerNeuralNetwork
from ReadExternalData import ReadIrisData

(X, Y) = ReadIrisData()

factor = 1/2
netwerk = MultiLayerNeuralNetwork([7, 5, 3], X, Y, factor)
netwerk.train(2500, 0.01)
