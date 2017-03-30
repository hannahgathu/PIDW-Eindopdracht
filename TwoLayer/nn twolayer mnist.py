import autograd.numpy as np
from autograd import grad
from TwoLayerNeuralNetwork import TwoLayerNeuralNetwork
from read_ubyte import ReadImages
from read_ubyte import ReadLabels

Y = 1/9 * ReadLabels('t10k-labels.idx1-ubyte')
X = 255 * ReadImages('t10k-images.idx3-ubyte')

n = TwoLayerNeuralNetwork(100, X, Y, 1, 1)
print("Training...")
n.train(10, 0.01)
print("Klaar")
uit = np.zeros(np.shape(X)[0])
verschil = np.zeros(np.shape(X)[0])
for i in range(np.shape(X)[0]):
    uit[i] = n.predict(X[i])
    verschil[i] = round(9 * uit[i]) - round(9 * Y[i])
print("Er zijn", np.count_nonzero(verschil), "foute antwoorden gegeven.")
print(verschil)
