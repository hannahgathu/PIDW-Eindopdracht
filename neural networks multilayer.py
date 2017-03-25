""" Begin van meerdere lagen
(zie bijgevoegde afbeelding?)

geschreven voor twee invoer en twee verborgen eenheden
dus aantal connecties en aantal gewichten = 2 (n+1) = 6

KLAAR:
sigmoid
predict
misfit

uitbreiden naar meer input (nog niet getest)
uitbreiden naar meer verborgen eenheden (nog niet getest)


BIJNA KLAAR:
gradient
train (probleem: autograd werkt nog niet)

in dit bestand wordt autograd gebruikt (installatie: pip install autograd)
"""
#import numpy as np
import autograd.numpy as np # om autograd te kunnen gebruiken
from autograd import grad


## test data
# importeren van IrisData
# symbolen zijn dezelfde als in readIrisData.py
# de volgende regel voert alles in readIrisData.py uit
#from readIrisData import *
#X = X/10
#y = y/3

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0, 0, 0, 1]) # AND

print("Invoer:      \n", X)
print("Test uitvoer:\n", Y)

## class
class MultilayerNeuralNetwork():
    """ Neuraal netwerk met één verborgen laag """
    def __init__(self, aantal, parameters):
        self.aantal = aantal # aantal eenheden in de verborgen laag
        self.parameters = parameters # lijst met gewichten en bias
        self.gewichten = self.parameters[:-(self.aantal+1)] # lijst met gewichten voor alle connecties
        self.bias = self.parameters[-(self.aantal+1):] # lijst met bias voor alle eenheden

    # functies
    def sigmoid(self, x):
        """ activatiefunctie sigma """
        return 1/(1+np.exp(-x))

    def misfit(self, uitkomst, voorbeeld):
        """ bereken het verschil met de test uitvoer en kwadrateer deze """
        return (uitkomst - voorbeeld)**2

    def gradient(self, function):
        """ bereken de gradient van een functie met behulp van autograd """
        return grad(function)

    def output(self, invoer):
        """ voorspelt een uitvoer voor de gegeven invoer
        verbeteren door een meer algemene code te maken --> zie predict
        """
        # bepaal tussen uitkomsten (van 2 verborgen eenheden)
        # uitbreiden door: for i in range(self.aantal): ?
        #print("Tussenstappen functie 'output':")
        s1 = np.dot(invoer, self.gewichten[0:2]) + self.bias[0] # 'transpose' om de goede vorm te krijgen
        #print(s1)
        y1 = self.sigmoid(s1)
        #print(y1)
        s2 = np.dot(invoer, self.gewichten[2:4]) + self.bias[1]
        #print(s2)
        y2 = self.sigmoid(s2)
        #print(y2)
        y = np.array([y1, y2])
        #print(y)

        # bepaal einduitkomst
        #s3 = np.dot(y, self.gewichten[4,6]) + self.bias[2]
        s3 = y1 * self.gewichten[4] + y2 * self.gewichten[5] + self.bias[2]
        #print(s3)
        y3 = self.sigmoid(s3)
        #print(y3)
        return y3

    def predict(self, invoer):
        """ voorspelt een uitvoer voor de gegeven invoer """
        # bepaal tussen uitkomsten
        temp_w = self.gewichten.copy()
        uit = np.zeros(self.aantal) # dit is een lijst met de uitvoer van de verborgen eenheden
        for eenheid in range(self.aantal):
            #low = len(invoer) * eenheid # eerste waarde
            #high = len(invoer)] * (eenheid + 1) # laatste waarde
            #s = np.dot(invoer, self.gewichten[low:high]) + self.bias[eenheid]
            s = np.dot(invoer, temp_w[0:len(invoer)]) + self.bias[eenheid]
            temp_w = temp_w[len(invoer):]
            y = self.sigmoid(s)
            uit[eenheid] = y

        # bepaal einduitkomst
        s_uit = self.bias[-1] + np.dot(uit, temp_w)
        return self.sigmoid(s_uit)

    def train(self, test_in, test_uit, iteraties, alpha):
        """ train netwerk met gegeven invoer en gegeven uitvoer
        optimaliseren door inproducten te gebruiken in plaats van for-loops
        """
        for i in range(iteraties):
            fout = 0
            for j in range(len(test_uit)): # of test_in
                # bepaal uitvoer en fout voor elke invoer
                uitvoer = self.predict(test_in[j])
                fout = fout + self.misfit(uitvoer, test_uit[j])
            g = self.gradient(fout)
            self.parameters = self.parameters - alpha * 1
            print(g)


# w heeft lengte 6, b heeft lengte 3
#w = np.random.random(6) # 6 willekeurige getallen tussen 0 en 1
#b = np.random.random(3)
p = np.random.random(9)

netwerk = MultilayerNeuralNetwork(2, p)

print("\nTest functies: ")
print("Oude parameters: ", netwerk.parameters)
for row in X:
    uit = netwerk.output(row)
    print("In:", row, " Uit:", uit)

netwerk.train(X, Y, 100, 1e0)
'''
print("Nieuwe parameters:", netwerk.parameters)
for row in X:

    uit = netwerk.predict(row)
    print("In:", row, " Uit:", uit)
'''
