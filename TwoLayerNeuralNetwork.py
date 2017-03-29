import autograd.numpy as np # om autograd te kunnen gebruiken
from autograd import grad

class TwoLayerNeuralNetwork():
    """ Neuraal netwerk met één verborgen laag:
        k is het aantal eenheden in de verborgen laag
        X is de invoer van data om te leren,
        Y is de bekende uitvoer bij de testdata
        n is het getal waarmee X genormaliseerd wordt
        m is het getal waarmee Y genormaliseerd wordt
        """
    def __init__(self, k, X, Y, n, m):
        self.l = np.shape(X)[-1] #lengte van 1 input
        self.j = np.size(Y) #aantal voorbeelden
        w_len = k * (self.l + 1)
        b_len = k + 1
        self.k = k
        #p is een lijst met beginschattingen voor de gewichten [w1,w2,...wn,b]
        self.p = np.random.random(w_len + b_len)
        self.X = np.round(X/n, 1)
        self.Y = np.round(Y/m, 1)
        self.m = m
        self.output = np.zeros((self.j, self.l + 2))
        print('output: ',self.output)
        self.printbegin()

    def printbegin(self):
        print("Test invoer:\n", self.X)
        print("Test uitvoer:\n", np.round(self.m *self.Y, 0))
        print("\nOude parameters:  \n", self.p)
        for i in range(self.j):
            self.output[i][0:self.l] = self.X[i]
            self.output[i][self.l] = self.predict(self.X[i])
            
    # functies
    def sigma(self, x):
        """ activatiefunctie sigma """
        return 1/(1+np.exp(-x))

    def fout(self, p):
        """ bepaalt de fout met als variabelen de gewichten en bias
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
        print('\n Train')
        for i in range(iteraties):
            gradient_functie = grad(self.fout)
            gradient = gradient_functie(self.p)
            self.p = self.p - alfa * gradient
        self.printeind()
        
    def printeind(self):        
        print("Nieuwe parameters: \n", self.p)
        for i in range(self.j):
            self.output[i][self.l + 1] = self.predict(self.X[i])
            print('In: {0}, Oud uit: {1:.0f}, Nieuw uit: {2:.0f}'.format(self.output[i][0:self.l], self.output[i][self.l], self.output[i][self.l+1]))
            
    def predict(self, invoer):
        """ voorspelt een uitvoer voor de gegeven invoer """
        return np.round(self.m*self.bereken_y(invoer, self.p))


    
