import autograd.numpy as np # om autograd te kunnen gebruiken
from autograd import grad

class TwoLayerNeuralNetwork():
    """ Neuraal netwerk met één verborgen laag:
        k is het aantal eenheden in de verborgen laag
        X is de invoer van data om te leren,
        Y is de bekende uitvoer bij de testdata
        """
    def __init__(self, k, X, Y):
        w_len = k * (np.shape(X)[-1] + 1)
        b_len = k + 1
        self.k = k
        #p is een lijst met beginschattingen voor de gewichten [w1,w2,...wn,b]
        self.p = np.random.random(w_len + b_len)
        self.X = X 
        self.Y = Y 

    # functies
    def sigma(self, x):
        """ activatiefunctie sigma """
        return 1/(1+np.exp(-x))

    def fout(self, p):
        """ bepaalt de fout met als variabelen de gewichten en bias
        bewerkingen zijn ongeveer gelijk aan die in self.predict()
        """
        y_uit = self.bereken_y(self.X, p)
        verschil = y_uit - self.Y
        return np.dot(verschil, verschil)

    def bereken_y(self, invoer, p):
        w = p[:-(self.k+1)]
        b = p[-(self.k+1):]

        s_uit = b[-1]
        for eenheid in range(self.k):
            wi = w[0:np.shape(invoer)[-1]]
            si = np.dot(invoer, wi) + b[eenheid]
            w = w[np.shape(invoer)[-1]:]
            yt = self.sigma(si)
            s_uit = s_uit + w[-self.k+eenheid] * yt

        # einduitkomst
        return self.sigma(s_uit)

    def train(self, iteraties, alfa):
        """ train netwerk met gegeven invoer en gegeven uitvoer """
        for i in range(iteraties):
            gradient_functie = grad(self.fout)
            gradient = gradient_functie(self.p)
            self.p = self.p - alfa * gradient
            
    def predict(self, invoer):
        """ voorspelt een uitvoer voor de gegeven invoer """
        return self.bereken_y(invoer, self.p)


    
