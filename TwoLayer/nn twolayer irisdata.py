import autograd.numpy as np
from autograd import grad
import csv
from TwoLayerNeuralNetwork import TwoLayerNeuralNetwork

m = 150
label = {"Iris-setosa" : 1, "Iris-versicolor" : 2, "Iris-virginica" : 3}
X = np.zeros((m,4))
Y = np.zeros(m)

with open('Iris.csv', newline='') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     #skip headers
     next(reader, None)

     #loop over rows
     for row in reader:
         #index
         k = int(row[0])-1
         # features
         X[k,:] = row[1:5]
         #labels
         Y[k] = label[row[5]]

# Now, X contains 150 input samples with 4 features each:
#(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
# and y contains 150 labels (1, 2, 3)
# Dit is de trainingsdata

netwerk = TwoLayerNeuralNetwork(3, X, Y)
netwerk.train(10000, 0.1)
