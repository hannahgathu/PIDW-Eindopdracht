""" test bestand voor mnist-data """

import autograd.numpy as np
from autograd import grad
from MultiLayerNeuralNetwork import MultiLayerNeuralNetwork
from ReadExternalData import ReadImages
from ReadExternalData import ReadLabels

TrainImages = ReadImages('train-images.idx3-ubyte', 60000)
TrainLabels = ReadLabels('train-labels.idx1-ubyte', 60000)

TestImages = ReadImages('t10k-images.idx3-ubyte', 10000)
TestLabels = ReadLabels('t10k-labels.idx1-ubyte', 10000)

# deze kunnen worden gevarieerd:
k = [9,5]
aantal = 5000

# dit zijn werkende waarden,
# bedoeld als hulpmiddel bij het vinden van goede parameters:
factor = 1/(2 * sum(k))
alfa = len(k) / (aantal * sum(k))

n = MultiLayerNeuralNetwork(k, TrainImages[:aantal], TrainLabels[:aantal], factor)

print("\nMaximale initiële weging is {}".format(factor))

n.train(5000, alfa)

n.bepaal_succes(TrainImages, TrainLabels)
n.bepaal_succes(TestImages, TestLabels)

print("test eerste 15:")
for i in range(15):
    print("Uit:",n.predict(TrainImages[i]), " Verwacht:",TrainLabels[i])
