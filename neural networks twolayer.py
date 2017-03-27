""" Begin van meerdere lagen
(zie bijgevoegde afbeelding?)

# één verborgen laag werkt

KLAAR:
sigmoid
predict
fout
train (+ gradient)
uitbreiden naar meer input
uitbreiden naar meer verborgen eenheden

TODO:
uitbreiden naar meer lagen

opruimen van niet nuttige code?
optimaliseren?

in dit bestand wordt autograd gebruikt (installatie: pip install autograd)
zie ook https://github.com/HIPS/autograd
"""

import autograd.numpy as np # om autograd te kunnen gebruiken
from autograd import grad
from TwoLayerNeuralNetwork import TwoLayerNeuralNetwork


## test data
# importeren van IrisData
# symbolen zijn dezelfde als in readIrisData.py
# de volgende regel voert alles in readIrisData.py uit
#from readIrisData import *
#X = X/10
#y = y/3

# invoer -> 2
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0, 0, 0, 1]) # OR

# invoer -> 3
X = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
#Y = np.array([0, 1, 1, 1, 1, 1, 1, 1]) # minimaal 1
Y = np.array([0, 0, 0, 1, 0, 1, 1, 1])# minimaal 2

print("Invoer:      \n", X)
print("Test uitvoer:\n", Y)


# op dit moment zijn de invoer van TwoLayerNeuralNetwork():
# aantal verborgen eenheden; test_input; test_output.
netwerk = TwoLayerNeuralNetwork(10, X, Y)

print("\n")
print("Oude parameters:  ", netwerk.p)
for row in X:
    uit = netwerk.predict(row)
    print("In:", row, " Uit:", uit)

print("\nTrain")
netwerk.train(100, 1e0)

print("Nieuwe parameters:", netwerk.p)
for row in X:
    uit = netwerk.predict(row)
    print("In:", row, " Uit:", uit)
