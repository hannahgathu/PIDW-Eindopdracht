import autograd.numpy as np # om autograd te kunnen gebruiken
from autograd import grad

class TwoLayerNeuralNetwork():
    """ Neuraal netwerk met één verborgen laag:
        k is het aantal eenheden in de verborgen laag
        test_in is de invoer van data om te leren,
        test_uit is de bekende uitvoer bij de testdata"""
    def __init__(self, k, test_in, test_uit):
        w_len = k * (np.size(test_uit) + 1)
        b_len = k + 1
        p = np.random.random(w_len + b_len)

        self.k = k 
        self.p = p # lijst met gewichten en bias
        self.w = self.p[:-(self.k+1)] # lijst met gewichten voor alle connecties
        self.bias = self.p[-(self.k+1):] # lijst met bias voor alle eenheden
        self.test_in = test_in 
        self.test_uit = test_uit 


    # functies
    def sigmoid(self, x):
        """ activatiefunctie sigma """
        return 1/(1+np.exp(-x))


    def fout(self, variabelen):
        """ bepaalt de fout met als variabelen de gewichten en bias
        bewerkingen zijn ongeveer gelijk aan die in self.predict()
        """
        weging = variabelen[:-(self.aantal+1)]
        bias = variabelen[-(self.aantal+1):]

        s_uit = bias[-1]
        for eenheid in range(self.aantal):
            w = weging[0:np.shape(self.test_in)[-1]]
            x = np.transpose(self.test_in)
            s = np.dot(w, x) + bias[eenheid]
            weging = weging[np.shape(self.test_in)[-1]:]
            y = self.sigmoid(s)
            #tussen[eenheid,:] = y
            s_uit = s_uit + weging[-self.aantal+eenheid] * y

        # einduitkomst
        y_uit = self.sigmoid(s_uit)
        verschil = y_uit - self.test_uit
        return np.dot(verschil, verschil)


    def predict(self, invoer):
        """ voorspelt een uitvoer voor de gegeven invoer """
        weging = self.parameters[:-(self.aantal+1)]
        bias = self.parameters[-(self.aantal+1):]

        s_uit = bias[-1]
        for eenheid in range(self.aantal):
            w = weging[0:np.shape(invoer)[-1]]
            x = np.transpose(invoer)
            s = np.dot(w, x) + bias[eenheid]
            weging = weging[np.shape(invoer)[-1]:]
            y = self.sigmoid(s)
            #tussen[eenheid,:] = y
            s_uit = s_uit + weging[-self.aantal+eenheid] * y

        # einduitkomst
        y_uit = self.sigmoid(s_uit)
        return(y_uit)


    def train(self, iteraties, alpha):
        """ train netwerk met gegeven invoer en gegeven uitvoer """
        for i in range(iteraties):
            gradient_functie = grad(self.fout)
            gradient = gradient_functie(self.parameters)
            self.parameters = self.parameters - alpha * gradient

