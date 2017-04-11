import autograd.numpy as np # om autograd te kunnen gebruiken
from autograd import grad
import math

class MultiLayerNeuralNetwork():
    """ Neuraal netwerk met één verborgen laag:
        X is een lijst van lijsten, dit is de invoer van data om te leren
        f is het aantal features per invoer(dus de lengte van de lijsten in  X)
        l is het aantal lagen (in dit geval 2)
        W is een lijst met met alle w_i's
        w_i is een matrix met de gewichten van de connecties in laag i
        B is een lijst met de biassen van alle neuronen
        Y is de bekende uitvoer bij de testdata
        v is het aantal voorbeelden
        n is het getal waarmee X genormaliseerd wordt
        m is het getal waarmee Y genormaliseerd wordt
        k is een lijst met het aantal neuronen per laag
        """
    def __init__(self, k, X, Y):
        self.n = np.max(X)
        self.X = np.round(X/self.n, 1)
        self.f = np.shape(X)[-1] #lengte van 1 input
        self.l=len(k) #aantal lagen
        self.v = np.size(Y) #aantal voorbeelden
        self.k = k
        W=[]               
        self.m = np.max(Y)
        self.Y = np.round(Y/self.m, 1)
        self.B=np.random.random(sum(self.k)+1)

        for i in range (len(k)-1):
            W.append(np.random.rand(k[i],self.f))
            
        W.append(np.random.random(k[-1])
        self.output = np.zeros((self.v, self.f + 3))
        self.printbegin()

    def printbegin(self):
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
        s_uit = p[-1]
        off = 0 #plek in p waar de wegingen voor deze laag beginnen
        offb = 0 #plek in p waar de biassen voor deze laag beginnen
        for i in range(len(self.k)):            
            for eenheid in range(self.k[i+1]):
                wi = p[off+eenheid*self.l:off+(eenheid+1)*self.l] # gewichten
                si = np.dot(invoer, wi) + p[-(sum(self.k)+ 1)+offb+eenheid] # inp + bias
                yt = self.sigma(si)
                s_uit = s_uit + p[-(2*self.k+1)+eenheid] * yt # totaal + weging * y
            off += self.k[i] * self.k[i+1]
            offb += k[i+1]
        return self.sigma(s_uit)

    def train(self, iteraties, alfa):
        """ train netwerk met gegeven invoer en gegeven uitvoer """
        for i in range(iteraties):
            gradient_functie = grad(self.fout)
            gradient = gradient_functie(self.p)
            self.p = self.p - alfa * gradient
        self.printeind()
                
    def printeind(self):        
        print("Nieuwe parameters na training: \n", self.p)
        print('Resultaten per voorbeeld:')
        aantal_fout = 0
        for i in range(self.j):
            self.output[i][self.l + 1] = self.predict(self.X[i])
            self.output[i][self.l + 2] = np.round(self.m * self.Y[i]) \
                                         - self.output[i][self.l +1]
            if self.output[i][self.l + 2] != 0:
                aantal_fout += 1
            #print('In: {0}, Oud uit: {1:.0f}, Nieuw uit: {2:.0f}, Verschil nieuw en correct: {3:.0f}'.\
                  #format(self.output[i][0:self.l], self.output[i][self.l], self.output[i][self.l+1], self.output[i][self.l+2]))
        print('Aantal verkeerd voorspelde antwoorden', aantal_fout)
                                            
    def predict(self, invoer):
        """ voorspelt een uitvoer voor de gegeven invoer """
        return np.round(self.m*self.bereken_y(invoer, self.p))
    
