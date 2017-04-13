""" test bestand voor mnist-data """

import autograd.numpy as np
from autograd import grad
from MultiLayerNeuralNetwork import MultiLayerNeuralNetwork
from read_ubyte import ReadImages
from read_ubyte import ReadLabels

TrainLabels = ReadLabels('train-labels.idx1-ubyte', 60000)
TrainImages = ReadImages('train-images.idx3-ubyte', 60000)

TestLabels = ReadLabels('t10k-labels.idx1-ubyte', 10000)
TestImages = ReadImages('t10k-images.idx3-ubyte', 10000)

n = MultiLayerNeuralNetwork([50,30,10], TrainImages, TrainLabels)
n.train(10, 0.01)
n.bepaal_succes(TrainImages, TrainLabels)
n.bepaal_succes(TestImages, TestLabels)
