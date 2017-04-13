""" MultiLayerNeuralNetwork
Klaar:
create_k
create_w
create_b
print_init
sigma

"""

import autograd.numpy as np # om autograd te kunnen gebruiken
from autograd import grad

class MultiLayerNeuralNetwork():
    """ Neuraal netwerk met één verborgen laag:
        k is een lijst van de aantallen eenheden per verborgen laag
        X is de invoer van data om te leren,
        Y is de bekende uitvoer bij de testdata
        n is het getal waarmee X genormaliseerd wordt
        m is het getal waarmee Y genormaliseerd wordt
        """
    def __init__(self, k=None, X=None, Y=None):
        if k is None: k = 1
        if X is None: X = np.array([1,1])
        if Y is None: Y = np.array([1])

        self.l = np.shape(X)[-1] #lengte van 1 input
        self.j = np.size(Y) #aantal voorbeelden
        self.k = self.create_k(k)
        self.w = self.create_w()
        self.b = self.create_b()
        self.p = [self.w, self.b]
        self.n = np.max(X)
        self.m = np.max(Y)
        self.X = np.round(X/self.n, 1)
        self.Y = np.round(Y/self.m, 1)
        # self.output = np.zeros((self.j, self.l + 3))
        # self.printbegin()
        self.print_init()

    def __str__(self):
        t = ''
        t = t + "Neuraal netwerk heeft {} verborgen lagen. ".format(len(self.k))
        t = t + "De lagen bevatten respectivelijk {} neuronen. ".format(self.k)
        t = t + "De testdata bestaat uit {} voorbeelden met {} waarden.\n".format(self.j, self.l)
        return t

    def create_k(self, k):
        """ Zet 'k' in een goede vorm (lijst) """
        if isinstance(k, int):
            return [k]
        return k

    def create_w(self):
        """ Maak een lijst met arrays met gewichten.
        De arrays representeren connecties tussen verschillende lagen.
        """
        q = [self.l] + self.k + [1]
        w = []
        for i in range(len(q)-1):
            w_i = np.random.random([q[i], q[i+1]])
            w.append(w_i)
        return w

    def create_b(self):
        """ Maak een lijst met arrays met biassen.
        De arrays zijn vectoren met de biassen van één laag.
        """
        q = self.k + [1]
        b = []
        for i in range(len(q)):
            b_i = np.random.random(q[i])
            b.append(b_i)
        return b

    def export_parameters(self, bestand):
        """ Exporteer parameters """
        # voor een correcte werking zouden alle self.* hierin moeten
        print("Bezig met exporteren van netwerk eigenschappen")
        if not isinstance(bestand, str):
            bestand = str(bestand)
        np.savez_compressed(str(bestand)+'.npz',
                            l=self.l, j=self.j, k=self.k, w=self.w, b=self.b,
                            p=self.p, n=self.n, m=self.m, X=self.X, Y=self.Y
                            )
        print("Exporteren voltooid\n")

    def import_parameters(self, bestand):
        """ Importeer parameters uit een .npz bestand, gecreëerd door export_parameters() """
        print("Bezig met laden van data in " +str(bestand) +".npz")
        if not isinstance(bestand, str):
            bestand = str(bestand)
        data = np.load(bestand+'.npz')
        self.l, self.j, self.k, self.w, self.b = data['l'], data['j'], data['k'], data['w'], data['b']
        self.p, self.n, self.m, self.X, self.Y = data['p'], data['n'], data['m'], data['X'], data['Y']
        print("Het netwerk heeft nieuwe parameters:")
        print(self)

    def print_init(self):
        print("Nieuw neuraal netwerk gemaakt met {} verborgen lagen.".format(len(self.k)),
              "De lagen bevatten respectivelijk {} neuronen.".format(self.k),
              "De testdata bestaat uit {0} voorbeelden met {1} waarden.\n".format(self.j, self.l))
    # def printbegin(self):
    #     print("\nOude parameters:  \n", self.p)
    #     for i in range(self.j):
    #         self.output[i][0:self.l] = self.X[i]
    #         self.output[i][self.l] = self.predict(self.X[i])

    def sigma(self, x):
        """ activatiefunctie sigma """
        return 1/(1+np.exp(-x))

    # def fout(self, p):
    #     """ bepaalt de fout met als variabelen de gewichten en bias
    #     """
    #     y_uit = self.bereken_y(self.X, p)
    #     verschil = y_uit - self.Y
    #     return np.dot(verschil, verschil)

    # def bereken_y(self, invoer, p):
    #     s_uit = p[-1]
    #     off = 0 #plek in p waar de wegingen voor deze laag beginnen
    #     offb = 0 #plek in p waar de biassen voor deze laag beginnen
    #     for i in range(len(self.k)):
    #         for eenheid in range(self.k[i+1]):
    #             wi = p[off+eenheid*self.l:off+(eenheid+1)*self.l] # gewichten
    #             si = np.dot(invoer, wi) + p[-(sum(self.k)+ 1)+offb+eenheid] # inp + bias
    #             yt = self.sigma(si)
    #             s_uit = s_uit + p[-(2*self.k+1)+eenheid] * yt # totaal + weging * y
    #         off += self.k[i] * self.k[i+1]
    #         offb += k[i+1]
    #     return self.sigma(s_uit)

    # def train(self, iteraties, alfa):
    #     """ train netwerk met gegeven invoer en gegeven uitvoer """
    #     for i in range(iteraties):
    #         gradient_functie = grad(self.fout)
    #         gradient = gradient_functie(self.p)
    #         self.p = self.p - alfa * gradient
    #     self.printeind()

    # def printeind(self):
    #     print("Nieuwe parameters na training: \n", self.p)
    #     print('Resultaten per voorbeeld:')
    #     aantal_fout = 0
    #     for i in range(self.j):
    #         self.output[i][self.l + 1] = self.predict(self.X[i])
    #         self.output[i][self.l + 2] = np.round(self.m * self.Y[i]) \
    #                                      - self.output[i][self.l +1]
    #         if self.output[i][self.l + 2] != 0:
    #             aantal_fout += 1
    #         #print('In: {0}, Oud uit: {1:.0f}, Nieuw uit: {2:.0f}, Verschil nieuw en correct: {3:.0f}'.\
    #               #format(self.output[i][0:self.l], self.output[i][self.l], self.output[i][self.l+1], self.output[i][self.l+2]))
    #     print('Aantal verkeerd voorspelde antwoorden', aantal_fout)
    #
    # def predict(self, invoer):
    #     """ voorspelt een uitvoer voor de gegeven invoer """
    #     return np.round(self.m*self.bereken_y(invoer, self.p))


if __name__ == "__main__":
    """ basis testen voor het netwerk """
    netwerk = MultiLayerNeuralNetwork([4,5], np.array([[2,4,6],[3,5,7]]), np.array([1,2]))
    netwerk.export_parameters("test")
    # print("w:", [np.shape(netwerk.w[a]) for a in range(len(netwerk.w))], netwerk.w)
    # print("b:", [np.shape(netwerk.b[a]) for a in range(len(netwerk.w))], netwerk.b)
    n = MultiLayerNeuralNetwork()
    n.import_parameters("test")

# MultiLayerNeuralNetwork([3,4,5,6,7,8], np.array([[2,4,6],[3,5,7]]), np.array([1,2]))
# MultiLayerNeuralNetwork(2, np.array([[2,4,6],[3,5,7]]), np.array([1,2]))
