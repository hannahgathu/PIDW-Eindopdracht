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
    def __init__(self, k, X, Y, n, m):
        self.l = np.shape(X)[-1] #lengte van 1 input
        self.j = np.size(Y) #aantal voorbeelden
        w_len = k * (self.l + 1)
        b_len = k + 1
        self.k = k
        #p is een lijst met beginschattingen voor de gewichten [w1,w2,...wn,b]
        self.p = np.random.random(w_len + b_len)
        self.m = np.max(Y)
        self.n = np.max(X)
        self.X = np.round(X/self.n, 1)
        self.Y = np.round(Y/self.m, 1)
        self.output = np.zeros((self.j, self.l + 3))
        # self.printbegin()

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

    def bereken_y_oud(self, invoer, p):
        s_uit = p[-1]
        for eenheid in range(self.k):
            wi = p[eenheid*self.l:(eenheid+1)*self.l] # gewichten
            si = np.dot(invoer, wi) + p[-(self.k+1)+eenheid] # inp + bias
            yt = self.sigma(si)
            s_uit = s_uit + p[-(2*self.k+1)+eenheid] * yt # totaal + weging * y
        return self.sigma(s_uit)

    def bereken_y(self, invoer, p):
        w = p[:-(2*self.k+1)] # neem alle gewichten
        w = np.reshape(w, [self.l, -1]) # maak een matrix
        d = np.dot(invoer, w) # matrix product invoer en w
        b = np.tile(p[-(self.k+1):-1], (np.shape(d)[0],1)) # bias
        s = b + d
        s_uit = np.sum(s, axis=1) + p[-1]
        return self.sigma(s_uit)

    def train(self, iteraties, alfa):
        """ train netwerk met gegeven invoer en gegeven uitvoer """
        for i in range(iteraties):
            gradient_functie = grad(self.fout)
            gradient = gradient_functie(self.p)
            self.p = self.p - alfa * gradient
            if i%round(0.1*iteraties) is 0:
                print(i, "/", iteraties, "iteraties gedaan") # print voortgang
        print(iteraties, "/", iteraties, "iteraties gedaan") # print voortgang
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
            #print('In: {0}, Oud uit: {1:.0f}, Nieuw uit: {2:.0f}, Verschil nieuw en correct: {3:.0f}'.\
                  #format(self.output[i][0:self.l], self.output[i][self.l], self.output[i][self.l+1], self.output[i][self.l+2]))
        print('Aantal verkeerd voorspelde antwoorden', aantal_fout)

    def predict(self, invoer):
        """ voorspelt een uitvoer voor de gegeven invoer """
        return np.round(self.m*self.bereken_y(invoer, self.p))


if __name__ == "__main__":
    X = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    Y = np.array([0, 0, 0, 1, 0, 1, 1, 1])# minimaal 2

    netwerk = TwoLayerNeuralNetwork(10, X, Y, 1, 1)
    print("nieuw")
    t1 = perf_counter()
    print(netwerk.bereken_y(netwerk.X, netwerk.p))
    t2 = perf_counter()
    print("Tijd:", t2-t1)


    print("oud")
    t3 = perf_counter()
    print(netwerk.bereken_y_oud(netwerk.X, netwerk.p))
    t4 = perf_counter()
    print("Tijd:", t4-t3)
