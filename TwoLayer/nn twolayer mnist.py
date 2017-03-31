import autograd.numpy as np
from autograd import grad
from TwoLayerNeuralNetwork import TwoLayerNeuralNetwork
from read_ubyte import ReadImages
from read_ubyte import ReadLabels

TrainLabels = 1/9 * ReadLabels('train-labels.idx1-ubyte', 60000)
TrainImages = 1/255 * ReadImages('train-images.idx3-ubyte', 60000)

TestLabels = 1/9 * ReadLabels('t10k-labels.idx1-ubyte', 10000)
TestImages = 1/255 * ReadImages('t10k-images.idx3-ubyte', 10000)

n = TwoLayerNeuralNetwork(10, TrainImages, TrainLabels, 1, 1)

print("Training...")
n.train(50, 0.1)
print("Klaar")

print("Test traindata")
uit = np.zeros(np.shape(TrainImages)[0])
verschil = np.zeros(np.shape(TrainImages)[0])
for i in range(np.shape(TrainImages)[0]):
    uit[i] = n.predict(TrainImages[i])
    verschil[i] = round(9 * uit[i]) - round(9 * TrainLabels[i])
aantal_fout = np.count_nonzero(verschil)
print("Er zijn {0} foute antwoorden gegeven. ({1:.1f}%)".format(aantal_fout,
                                                         100*aantal_fout/np.size(verschil)))

print("\nTest testdata")
uit = np.zeros(np.shape(TestImages)[0])
verschil = np.zeros(np.shape(TestImages)[0])
for i in range(np.shape(TestImages)[0]):
    uit[i] = n.predict(TestImages[i])
    verschil[i] = round(9 * uit[i]) - round(9 * TestLabels[i])
aantal_fout = np.count_nonzero(verschil)
print("Er zijn {0} foute antwoorden gegeven. ({1:.1f}%)".format(aantal_fout,
                                                         100*aantal_fout/np.size(verschil)))
