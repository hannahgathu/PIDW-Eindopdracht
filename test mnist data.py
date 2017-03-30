import autograd.numpy as np
from autograd import grad
from TwoLayerNeuralNetwork import TwoLayerNeuralNetwork
from NeuralNetwork import NeuralNetwork
from read_ubyte import *


Y = Y/9
X = X/255
"""

Y = Y.astype('float16')
print(type(Y[1]))
"""

'''
# for row in X:
#     print(n.predict(row))
for t in range(100, 5001, 100):
    y = Y[:t]
    x = X[:t]
    n = TwoLayerNeuralNetwork(5, x, y)
    n.train(10, 1)
    print("t = ", t, " Output:", n.predict(X[1]))
'''

m = NeuralNetwork(X, Y)
m.train(1000, 0.01)
i = 0
for row in X:
    uit = m.predict(row)
    print("Uit:", uit*9, "Doel:", 9*Y[i])
    i += 1
