import autograd.numpy as np # om autograd te kunnen gebruiken
from autograd import grad
from time import perf_counter

class TwoLayerNeuralNetwork():
    """ Neuraal netwerk met één verborgen laag:
        k is het aantal eenheden in de verborgen laag
        X is de invoer van data om te leren,
        Y is de bekende uitvoer bij de testdata
        n is het getal waarmee X genormaliseerd wordt
        m is het getal waarmee Y genormaliseerd wordt
        """
    def __init__(self, k, X, Y):
        self.l = np.shape(X)[-1] #lengte van 1 input
        self.j = np.size(Y) #aantal voorbeelden
        w_len = k * (self.l + 1) #aantal connecties = aantal weigths
        b_len = k + 1 #aantal neuronen = aantal biassen
        self.k = k
        #p is een lijst met beginschattingen voor de gewichten [w1,w2,...wn,b]
        self.p = np.random.random(w_len + b_len)
        self.m = np.max(Y)
        self.n = np.max(X)
        self.X = np.round(X/self.n, 1)
        self.Y = np.round(Y/self.m, 1)
        self.output = np.zeros((self.j, self.l + 3))
        self.printinit()
        self.printbegin()

    def printinit(self):
        """ print een aantal eigenschappen van het netwerk """
        print("Nieuw neuraal netwerk gemaakt met 1 verborgen laag, met daarin {0} eenheden."
              .format(self.k))
        print("De testdata bestaat uit {0} voorbeelden met {1} waarden."
              .format(self.j, self.l))
        print()

    def printbegin(self):
        """Vult output met de oude uitkomsten"""
        for i in range(self.j):
            self.output[i][0:self.l] = self.X[i]
            self.output[i][self.l] = self.predict(self.X[i])

    def sigma(self, x):
        """ activatiefunctie sigma """
        return 1/(1+np.exp(-x))

    def fout(self, p):
        """ bepaal de fout met als variabelen de gewichten en bias """
        y_uit = self.bereken_y_oud(self.X, p)
        verschil = y_uit - self.Y
        return np.dot(verschil, verschil)

    def bereken_y_oud(self,invoer,p):
        """ bereken de uitvoer van het netwerk middels een forloop """
        s_uit = p[-1]
        for eenheid in range(self.k):
            wi = p[eenheid*self.l:(eenheid+1)*self.l] # gewichten
            si = np.dot(invoer, wi) + p[-(self.k+1)+eenheid] # inp + bias
            yt = self.sigma(si)
            s_uit = s_uit + p[-(2*self.k+1)+eenheid] * yt # totaal + weging * y
        return self.sigma(s_uit)


    def bereken_y_1(self, invoer, p):
        """ bereken de uitvoer van het netwerk """
        if len(np.shape(invoer)) is 1: # this makes a two-dim matrix from a list
            invoer = np.array([invoer])
        w = p[:-(2*self.k+1)] # neem alle gewichten van input naar laag
        w = np.reshape(w, [self.l, -1], 'F') # maak een matrix: rij<>l ; kol<>k
        d = np.dot(invoer, w) # matrix product invoer en w: rij<>j ; kol<>k
        b = np.tile(p[-(self.k+1):-1], (self.j, 1)) # maak array met bias vlnr
        y = self.sigma(b + d) # bepaal y = sigma(<x,w> + b) voor alle k en j
        yw = y * self.p[-(2*self.k+1):-(self.k+1)] # bepaal y * w`
        s_uit = p[-1] + np.sum(yw, axis=1) # sommeer rijen en voeg bias_out toe
        return self.sigma(s_uit)

    def train(self, iteraties, alfa):
        """ train netwerk met gegeven invoer en gegeven uitvoer """
        print("Het netwerk wordt nu '{} keer getraind', met alpha = {}.".format(iteraties, alfa))
        for i in range(iteraties):
            gradient_functie = grad(self.fout)
            gradient = gradient_functie(self.p)
            self.p = self.p - alfa * gradient
            if i%np.int(np.ceil(0.1*iteraties)) == 0:
                print(i, "/", iteraties, "iteraties gedaan") # print voortgang
        print(iteraties, "/", iteraties, "iteraties gedaan") # print voortgang
        print()
        self.printeind()

    def printeind(self):
        print("Nieuwe parameters na training: \n", self.p)
        print('Resultaten per voorbeeld:')
        aantal_fout = 0
        for i in range(self.j):
            self.output[i][self.l + 1] = self.predict(self.X[i])
            self.output[i][self.l + 2] = np.round(self.m * self.Y[i]) \
                                         - self.output[i][self.l +1]
            if self.output[i][-1] != 0:
                aantal_fout += 1
            print('In: {0}, Oud uit: {1:.0f}, Nieuw uit: {2:.0f}, Verschil nieuw en correct: {3:.0f}'.\
                  format(self.output[i][0:self.l], self.output[i][self.l], self.output[i][self.l+1], self.output[i][self.l+2]))
        print('Aantal verkeerd voorspelde antwoorden', aantal_fout)

    def predict(self, invoer):
        """ voorspel een uitvoer voor de gegeven invoer """
        #print(self.bereken_y_oud(invoer, self.p))
        return np.round(self.m*self.bereken_y_oud(invoer, self.p))


def main():
    print("Maak een neuraal netwerk:")

    testaantal, testlengte = 2, 4
    X = np.random.random([testaantal, testlengte])
    Y = np.random.random(np.shape(X)[0])

    netwerk = TwoLayerNeuralNetwork(5, X, Y)
    netwerk.train(100, 0.1)
    netwerk.predict(X[0])

if __name__ == "__main__":
    main()
