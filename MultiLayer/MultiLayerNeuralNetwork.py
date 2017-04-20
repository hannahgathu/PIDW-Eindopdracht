""" MultiLayerNeuralNetwork """
import autograd.numpy as np # om autograd te kunnen gebruiken
from autograd import grad

class MultiLayerNeuralNetwork():
    """ Neuraal netwerk met meerdere verborgen lagen:
        argumenten:
        k is een lijst van de aantallen eenheden per verborgen laag
        X is de invoer van data om te leren (testdata),
        Y is de bekende uitvoer bij de testdata
        f is de maximale waarde van de initiele gewichten

        attributen:
        l: lengte van een invoer
        j: aantal voorbeelden in testdata
        k: lijst met aantal eenheden per verborgen laag
        p: lijst met
            * arrays met gewichten van verbindingen tussen lagen (1e helft)
            * arrays met biassen van elke laag (2e helft)
        n: normalisatie factor voor invoer
        m: normalisatie factor voor uitvoer
        X: invoer van testdata
        Y: uitvoer van testdata
        """


    def __init__(self, k=1, X='and', Y='', f=1):
        if X is 'and': (X, Y) = self.create_xy_and()
        if X is 'or': (X, Y) = self.create_xy_or()
        if X is 'xor': (X, Y) = self.create_xy_xor()
        self.l = np.shape(X)[-1]
        self.j = np.size(Y)
        self.k = self.create_k(k)
        self.p = self.create_w(f) + self.create_b()
        self.n = np.max(X) # normalisatie X
        self.m = np.max(Y) # normalisatie Y
        self.X = X
        self.Y = Y

        print("Nieuw neuraal netwerk gemaakt.")
        print(self)


    def __str__(self):
        s = "Neuraal netwerk heeft {} verborgen laag/lagen. ".format(len(self.k))
        s = s + "De lagen bevatten respectievelijk {} neuronen. ".format(self.k)
        s = s + "De testdata bestaat uit {} voorbeelden met {} waarden ieder.\n"\
            .format(self.j, self.l)
        return s


    def create_xy_and(self):
        """ maak testdata om een 'and'-poort te simuleren """
        print("Het volgende netwerk kan een 'and'-poort simuleren")
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        Y = np.array([0,0,0,1])
        return (X, Y)


    def create_xy_or(self):
        """ maak testdata om een 'or'-poort te simuleren """
        print("Het volgende netwerk kan een 'or'-poort simuleren")
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        Y = np.array([0,1,1,1])
        return (X, Y)


    def create_xy_xor(self):
        """ maak testdata om een 'xor'-poort te simuleren """
        print("Het volgende netwerk kan een 'xor'-poort simuleren")
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        Y = np.array([0,1,1,0])
        return (X, Y)


    def create_k(self, k):
        """ Zet 'k' in een goede vorm (lijst) als het een getal is"""
        if isinstance(k, int):
            return [k]
        return k


    def create_w(self, f):
        """ Maakt een lijst met arrays (matrices) met gewichten.
        Iedere array representeert de connecties tussen twee opeenvolgende
        lagen.
        """
        factor = f
        #lijst met aantal neuronen per laag (incl. start en eind)
        q = [self.l] + self.k + [1]
        w = []
        for i in range(len(q)-1):
            w_i = factor * np.random.random([q[i], q[i+1]])
            w.append(w_i)
        return w


    def create_b(self):
        """ Maakt een lijst met arrays (vectoren) met biassen.
        Iedere arrays representeert de biassen van één laag.
        """
        q = self.k + [1]
        b = []
        for i in range(len(q)):
            # b_i = np.random.random(q[i])
            b_i = np.zeros(q[i])
            b.append(b_i)
        return b


    def export_parameters(self, bestand):
        """ Exporteer parameters """
        print("Bezig met exporteren van netwerk eigenschappen")

        if not isinstance(bestand, str):
            bestand = str(bestand)
        if bestand[-4:] == '.npz':
            naam = bestand
        else:
            naam = bestand+'.npz'

        np.savez_compressed(naam, l=self.l, j=self.j, k=self.k,
                            p=self.p, n=self.n, m=self.m, X=self.X, Y=self.Y)
        print("Exporteren naar " + str(naam) + " voltooid\n")


    def import_parameters(self, bestand):
        """ Importeer parameters uit een .npz bestand,
        welke is gecreëerd door export_parameters()
        """
        if not isinstance(bestand, str):
            bestand = str(bestand)
        if bestand[-4:] == '.npz':
            naam = bestand
        else:
            naam = bestand+'.npz'

        print("Bezig met laden van data in " + str(naam) + ".npz")

        data = np.load(naam)
        (self.l, self.j, self.k, self.p, self.n, self.m, self.X, self.Y) = (
            data['l'], data['j'], data['k'], data['p'],
            data['n'], data['m'], data['X'], data['Y'])

        print("Het netwerk heeft nieuwe parameters:")
        print(self)


    def nieuwe_testdata(self, invoer, uitvoer):
        """ voeg data toe aan bestaande testdata """
        print("De nieuwe data wordt toegevoegd")
        self.X = np.append(self.X, invoer, axis=0)
        self.Y = np.append(self.Y, uitvoer, axis=0)
        self.j = np.size(self.Y)
        self.n = np.max(self.X)
        self.m = np.max(self.Y)
        print("De testdata heeft nu {} voorbeelden\n".format(self.j))


    def reset_testdata(self, invoer, uitvoer):
        """ vervang bestaande testdata door nieuwe data """
        print("De testdata wordt vervangen door nieuwe data")
        self.X = invoer/self.n
        self.Y = uitvoer/self.m
        self.l = np.shape(self.X)[-1]
        self.j = np.size(self.Y)
        self.n = np.max(self.X)
        self.m = np.max(self.Y)
        print("De testdata heeft nu {} voorbeelden\n".format(self.j))


    def sigma(self, x):
        """ activatiefunctie sigma """
        return 1/(1+np.exp(-x))


    def fout(self, p):
        """ bepaal de fout met als variabelen de gewichten en bias """
        y_uit = self.bereken_y(self.X, p)
        verschil = y_uit - self.Y
        return np.dot(verschil, verschil)


    def bereken_y(self, invoer, p):
        """ bepaal de uitvoer bij de invoer
        hulp-functie bij fout()
        """
        w = p[:len(self.k) + 1]
        b = p[len(self.k) + 1:]
        y = invoer / self.n
        for laag in range(len(w)):
            s = np.dot(y, w[laag]) + b[laag]
            y = self.sigma(s)
        return np.reshape(y * self.m, -1)


    def train(self, iteraties, alfa):
        """ train netwerk met gegeven invoer en gegeven uitvoer """
        print("Het netwerk wordt nu getraind.\n"+
               "Het aantal iteraties is {}. ".format(iteraties)+
               "De train snelheid alfa is {}.\n".format(alfa))
        for i in range(iteraties):
            gradient_functie = grad(self.fout)
            gradient = gradient_functie(self.p)
            for j in range(len(self.p)):
                self.p[j] = self.p[j] - alfa * gradient[j]

            if i%np.int(np.ceil(0.1*iteraties)) == 0:
                num = str(i).rjust(len(str(iteraties))) + " / " + str(iteraties)
                fout = str(self.fout(self.p))
                print("{} iteraties gedaan. Huidige fout: {}".format(num,fout))

        print(iteraties, "/", iteraties, "iteraties gedaan. "+
              "Huidige fout: " + str(self.fout(self.p)) +"\n")
        if self.j < 25: self.printeind()
        else: self.bepaal_succes(self.X, self.Y)


    def printeind(self):
        output = np.zeros((self.j, self.l + 3))
        print('Resultaten per voorbeeld:')
        aantal_fout = 0
        for i in range(self.j):
            output[i][0:self.l] = np.round(self.X[i])
            output[i][self.l] = np.round(self.Y[i])
            output[i][self.l+1] = np.round(self.predict(self.X[i]))
            output[i][self.l+2] = np.round(self.Y[i]) - output[i][self.l+1]

            if output[i][self.l + 2] != 0:
                aantal_fout += 1

            print("In: {0}, ".format(output[i][0:self.l])+
                  "Verwacht: {0:.0f}, ".format(output[i][self.l])+
                  "Uit: {0:.0f}, ".format(output[i][self.l+1])+
                  "Verschil met correct: {0:.0f}".format(output[i][self.l+2]))
        print('Aantal verkeerd voorspelde antwoorden:', aantal_fout)


    def bepaal_succes(self, invoer, uitvoer):
        """ print het aantal verkeerd voorspelde antwoorden
        de invoer en uitvoer zijn niet-genormaliseerde getallen
        """
        if not isinstance(invoer, np.ndarray):
            invoer = [invoer]
        if not isinstance(uitvoer, np.ndarray):
            uitvoer = [uitvoer]

        a = np.round(self.predict(invoer))
        b = np.round(uitvoer)
        c = a - b

        aantal_fout = np.count_nonzero(c)
        print('Aantal verkeerd voorspelde antwoorden: {} '.format(aantal_fout) +
              '({0:.1f} %)\n'.format(100 * aantal_fout / np.size(uitvoer)))


    def predict(self, invoer):
        """ voorspel een niet-afgeronde uitkomst voor de gegeven invoer """
        p = self.p
        return self.bereken_y(invoer, p)




if __name__ == "__main__":
    n = MultiLayerNeuralNetwork([3,3], 'xor') # and-poort
    n.train(1000, 1)
