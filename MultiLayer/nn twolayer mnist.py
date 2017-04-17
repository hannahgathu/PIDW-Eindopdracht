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

factor = 1/10000
n = MultiLayerNeuralNetwork([5,5], TrainImages[:100], TrainLabels[:100], factor)
n.train(1000, 0.001)
n.bepaal_succes(TrainImages, TrainLabels)
n.bepaal_succes(TestImages, TestLabels)

print("test eerste 15:")
for i in range(15):
    print("Uit:",n.predict(TrainImages[i]), " Verwacht:",TrainLabels[i])
