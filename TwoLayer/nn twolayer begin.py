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
from copy import deepcopy

# invoer -> 2
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0, 0, 0, 1]) # OR

# invoer -> 3
X = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
#Y = np.array([0, 1, 1, 1, 1, 1, 1, 1]) # minimaal 1
Y = np.array([0, 0, 0, 1, 0, 1, 1, 1])# minimaal 2

netwerk = TwoLayerNeuralNetwork(10, X, Y, 1, 1)
netwerk.train(100, 1e0)

 

