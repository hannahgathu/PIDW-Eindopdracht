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
    """ Neuraal netwerk met meerdere verborgen lagen:
        k is een lijst van de aantallen eenheden per verborgen laag
        X is de invoer van data om te leren,
        Y is de bekende uitvoer bij de testdata
        """
    def __init__(self, k=1, X=np.array([1,1]), Y=np.array([1])):
        self.l = np.shape(X)[-1] #lengte van 1 input
        self.j = np.size(Y) #aantal voorbeelden testinput
        self.k = self.create_k(k)
        self.w = self.create_w()
##        self.w = [np.array([[0,1], [0,1]]), np.array([[1,0], [1,0]]), np.array([0,1])]
        self.b = self.create_b()
##        self.b = [np.array([1,2]), np.array([3,4]), np.array([5])]
        self.p = self.w + self.b
        self.n = np.max(X) #getal waarmee X genormaliseerd wordt
        self.m = np.max(Y) #getal waarmee Y genormaliseerd wordt
        self.X = np.round(X/self.n, 1)
        self.Y = np.round(Y/self.m, 1)
        print(self)

    def __str__(self):
        t = "Neuraal netwerk heeft {} verborgen la(a)g(en). ".format(len(self.k))
        t = t + "De lagen bevatten respectievelijk {} neuronen. ".format(self.k)
        t = t + "De testdata bestaat uit {} voorbeelden met {} waarden ieder.\n"\
            .format(self.j, self.l)
        return t

    def create_k(self, k):
        """ Zet 'k' in een goede vorm (lijst) als het een getal is"""
        if isinstance(k, int):
            return [k]
        return k

    def create_w(self):
        """ Maak een lijst met arrays (matrices) met gewichten.
        Iedere array representeert de connecties tussen twee opeenvolgende
        lagen.
        """
        q = [self.l] + self.k + [1] #lijst met aantal neuronen per laag (incl. start en eind)
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
                            n=self.n, m=self.m, X=self.X, Y=self.Y
                            )
        print("Exporteren voltooid\n")

    def import_parameters(self, bestand):
        """ Importeer parameters uit een .npz bestand, gecreëerd door export_parameters() """
        print("Bezig met laden van data in " +str(bestand) +".npz")
        if not isinstance(bestand, str):
            bestand = str(bestand)
        data = np.load(bestand+'.npz')
        self.l, self.j, self.k, self.w, self.b = data['l'], data['j'], data['k'], data['w'], data['b']
        self.n, self.m, self.X, self.Y = data['n'], data['m'], data['X'], data['Y']
        print("Het netwerk heeft nieuwe parameters:")
        print(self)

    def sigma(self, x):
        """ activatiefunctie sigma """
        return 1/(1+np.exp(-x))

    def fout(self, p):
         """ bepaalt de fout met als variabelen de gewichten en bias
         """
         y_uit = self.bereken_y(self.X, p)
         verschil = np.transpose(y_uit) - self.Y
         verschil = verschil.flatten()
         return np.dot(verschil, verschil)
        
    def bereken_y(self, invoer, p):
        w = p[:len(self.k) + 1]
        b = p[len(self.k) + 1:]
        yt = invoer
        for i in range(len(w)):
            s = np.dot(yt, w[i]) + b[i]
            yt = self.sigma(s)
        return yt

    def train(self, iteraties, alfa):
        """ train netwerk met gegeven invoer en gegeven uitvoer """
        print('oude waarden', self.p)
        for i in range(iteraties):
            if i%100 == 0:
                print(self.fout(self.p))
            gradient_functie = grad(self.fout)
            gradient = gradient_functie(self.p)
            for i in range(len(self.p)):
                self.p[i] = self.p[i] - alfa * gradient[i]
        print('nieuwe waarden',  self.p)
        self.printeind()

    def printeind(self):
        output = np.zeros((self.j, self.l + 3))
        print('Resultaten per voorbeeld:')
        aantal_fout = 0
        for i in range(self.j):            
            output[i][0:self.l] = np.round(self.n *self.X[i])
            output[i][self.l] = self.predict(self.X[i])
            output[i][self.l + 1] = np.round(self.m * self.Y[i]) \
                                        - output[i][self.l]
            if output[i][self.l + 1] != 0:
                aantal_fout += 1
            if self.j <= 100:
                print('In: {0}, Uit: {1:0f}, Verschil met correct: {2:.0f}'.\
                   format(output[i][0:self.l], output[i][self.l],\
                          output[i][self.l+1]))
        print('Aantal verkeerd voorspelde antwoorden', aantal_fout)
    
    def predict(self, invoer):
         """ voorspelt een uitvoer voor de gegeven invoer """
         return np.round(self.m*self.bereken_y(invoer, self.p))


if __name__ == "__main__":
    """ basis testen voor het netwerk """
    netwerk = MultiLayerNeuralNetwork([4,5], np.array([[2,4,6],[3,5,7]]), np.array([1,2]))
    #netwerk.export_parameters("test")
    # print("w:", [np.shape(netwerk.w[a]) for a in range(len(netwerk.w))], netwerk.w)
    # print("b:", [np.shape(netwerk.b[a]) for a in range(len(netwerk.w))], netwerk.b)
    #n = MultiLayerNeuralNetwork()
    #n.import_parameters("test")
    netwerk.train(100, 1)

# MultiLayerNeuralNetwork([3,4,5,6,7,8], np.array([[2,4,6],[3,5,7]]), np.array([1,2]))
# MultiLayerNeuralNetwork(2, np.array([[2,4,6],[3,5,7]]), np.array([1,2]))
