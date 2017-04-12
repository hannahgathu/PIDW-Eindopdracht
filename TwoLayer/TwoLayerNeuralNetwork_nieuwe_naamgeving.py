import autograd.numpy as np # om autograd te kunnen gebruiken
from autograd import grad
from time import perf_counter

class TwoLayerNeuralNetwork():
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
        """
    def __init__(self, k, X, Y):
        self.n = np.max(X)
        self.X = np.round(X/self.n, 1)
        self.f = np.shape(X)[-1] #lengte van 1 input
        self.l=2 #aantal lagen
        self.v = np.size(Y) #aantal voorbeelden
        w_0_len = k * self.f  #aantal connecties in laag 0 = aantal gewichten in laag 0
        w_1_len = k # aantal connecties in laatste laag (in dit geval laag 1)
        b_len = k + 1 #aantal neuronen = aantal biassen
        self.k = k 
        #W is een lijst met arrays met beginschattingen voor de gewichten in laag i [w_o,w_1]
        self.w_0 = np.random.rand(self.k,self.f)
        self.w_1 = np.random.random(self.k)
        self.B=np.random.random(b_len)
        self.W=[self.w_0,self.w_1,self.B]
        self.m = np.max(Y)
        self.Y = np.round(Y/self.m, 1)

        self.output = np.zeros((self.v, self.f + 3))
        #self.printinit()
        self.vul_output()

    def printinit(self):
        """ print een aantal eigenschappen van het netwerk """
        print("Nieuw neuraal netwerk gemaakt met 1 verborgen laag, met daarin {0} eenheden."
              .format(self.k))
        print("De testdata bestaat uit {0} voorbeelden met {1} waarden."
              .format(self.v, self.f))
        print()

    def vul_output(self):
        """Vult output met de oude uitkomsten"""
        for i in range(self.v):
            self.output[i][0:self.f] = self.X[i]
            self.output[i][self.f] = self.predict(self.X[i])

    def sigma(self, x):
        """ activatiefunctie sigma """
        return 1/(1+np.exp(-x))

    def fout(self, W):
        """ bepaal de fout met als variabelen de gewichten en bias """
        y_uit = self.bereken_y(self.X, W)
        #print(y_uit)
        verschil = y_uit - self.Y
        return np.dot(verschil, verschil)

    def bereken_y_oud(self,invoer,W):
        """ bereken de uitvoer van het netwerk middels een forloop """
        b_uit = W[-1][-1]
        for neuron in range(self.k):
            si = np.dot(invoer, W[0][neuron]) + W[-1][neuron] # inp + bias
            yt = self.sigma(si)
            b_uit = b_uit + W[1][neuron] * yt # totaal + weging * y
        return self.sigma(b_uit)

    def bereken_y(self,invoer,W):
        

    def train(self, iteraties, alfa):
        """ train netwerk met gegeven invoer en gegeven uitvoer """
        print("Het netwerk wordt nu '{} keer getraind', met alpha = {}.".format(iteraties, alfa))
        for i in range(iteraties):
            gradient_functie = grad(self.fout)
            gradient = gradient_functie(self.W)
            for j in range(self.l+1):
                self.W[j] = self.W[j] - alfa * gradient[j]
            if i%np.int(np.ceil(0.1*iteraties)) == 0:
                print(i, "/", iteraties, "iteraties gedaan") # print voortgang
        #print(iteraties, "/", iteraties, "iteraties gedaan") # print voortgang
        print()
        self.printeind()

    def printeind(self):
        #print("Nieuwe parameters na training: \n", self.p)
        print('Resultaten per voorbeeld:')
        aantal_fout = 0
        for i in range(self.v):
            self.output[i][self.f + 1] = self.predict(self.X[i])
            self.output[i][self.f + 2] = np.round(self.m * self.Y[i]) \
                                         - self.output[i][self.f +1]
            if self.output[i][-1] != 0:
                aantal_fout += 1
            print('In: {0}, Oud uit: {1:.0f}, Nieuw uit: {2:.0f}, Verschil nieuw en correct: {3:.0f}'.\
                  format(self.output[i][0:self.f], self.output[i][self.f], self.output[i][self.f+1], self.output[i][self.f+2]))
        print('Aantal verkeerd voorspelde antwoorden', aantal_fout)

    def predict(self, invoer):
        """ voorspel een uitvoer voor de gegeven invoer """
        #print("y:",self.bereken_y(invoer, self.W,self.B))
        return np.round(self.m*self.bereken_y(invoer, self.W))


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
