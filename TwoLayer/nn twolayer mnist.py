""" test bestand voor mnist-data """

import autograd.numpy as np
from autograd import grad
from TwoLayerNeuralNetwork import TwoLayerNeuralNetwork
from read_ubyte import ReadImages
from read_ubyte import ReadLabels

TrainLabels = ReadLabels('train-labels.idx1-ubyte', 60000)
TrainImages = ReadImages('train-images.idx3-ubyte', 60000)

TestLabels = ReadLabels('t10k-labels.idx1-ubyte', 10000)
TestImages = ReadImages('t10k-images.idx3-ubyte', 10000)

n = TwoLayerNeuralNetwork(2, TrainImages, TrainLabels)

print("Training...")
n.train(50, 0.01)
print("Klaar")

print("Test traindata")
uit = np.zeros(np.shape(TrainImages)[0])
verschil = np.zeros(np.shape(TrainImages)[0])
for i in range(np.shape(TrainImages)[0]):
    uit[i] = n.predict(TrainImages[i])
    verschil[i] = round(9 * uit[i]) - round(9 * TrainLabels[i])
fout = np.count_nonzero(verschil)
print("Er zijn {0} foute antwoorden gegeven. ({1:.1f}%)".format(fout, 100*fout/np.size(verschil)))

print("\nTest testdata")
uit = np.zeros(np.shape(TestImages)[0])
verschil = np.zeros(np.shape(TestImages)[0])
for i in range(np.shape(TestImages)[0]):
    uit[i] = n.predict(TestImages[i])
    verschil[i] = round(9 * uit[i]) - round(9 * TestLabels[i])
fout = np.count_nonzero(verschil)
print("Er zijn {0} foute antwoorden gegeven. ({1:.1f}%)".format(fout, 100*fout/np.size(verschil)))
