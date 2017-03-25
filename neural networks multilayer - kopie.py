import numpy as np
from NeuralNetwork import NeuralNetwork

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0, 0, 0, 1]) # AND

print("Invoer:      \n", X)
print("Test uitvoer:\n", Y)

class MultilayerNetwork(NeuralNetwork):    
    
    def __init__(self, k, n, m, w, b, p_eind ,iterations, alpha):
        """ Neuraal netwerk met één verborgen laag:
        k is het aantal eenheden in de verborgen laag,
        n is het aantal coordinaten in één invoer
        m is het aantal voorbeelden (len(X))
        w is een lijst met gewichten voor alle connecties (len = n*k),
        b is een lijst met bias voor alle eenheden (len = k),
        p_eind is een lijst met gewichten voor de output en de bias
            voor de output (len = k+1)"""
        self.k = k
        self.n = n
        self.m = m
        self.w = w
        self.b = b
        self.p_e = p_eind
        self.iterations = iterations
        self.alpha = alpha

    def multigrad(self, X, y):        
        """berekent de gradient van de errorfunctie voor multilayers"""
        #s is per eenheid een rij met inproduct per voorbeeld
        s = np.zeros((k,m))
        a = np.zeros((k,m))
        for l in range(k):
            s[l] = np.dot(X, self.w[l*n:(l+1)*n]) + b[l]
            a[l] = self.sigma(s[l])* self.p_e[l]
            
        #yt is per voorbeeld een waarde (som vd inproducten per eenheid)
        yt = a.sum(axis = 0) + self.p_e[-1]
        g = np.zeros(n+1)
        g[:-1] = 2 * np.dot(np.transpose(X)*yt*yt*(1-yt)*(yt-Y), np.ones(self.m))
        g[-1]  = 2*np.dot(yt*yt*(1 - yt)*(yt - Y),np.ones(self.m))        
        print('yt',yt)
        print('g', g)
        
                   

    def train(self, X, y):
        """X is de invoer, y de uitvoer"""
        for iteration in range(self.iterations):
            g = self.multigrad(X, y)
            self.p = self.p-self.alpha*g
        
    # functies
    def output(self, invoer):
        """ geeft een output (werkt zoals classify) """
        # bepaal tussen uitkomsten (van 2 verborgen eenheden)
        # uitbreiden door: for i in range(self.aantal): ?
        #print("Tussenstappen functie 'output':")
        s1 = np.dot(invoer, self.w[0:2]) + self.b[0] # 'transpose' om de goede vorm te krijgen
        #print(s1)
        y1 = self.sigma(s1)
        #print(y1)
        s2 = np.dot(invoer, self.w[2:4]) + self.b[1]
        #print(s2)
        y2 = self.sigma(s2)
        #print(y2)
        y = np.array([y1, y2])
        #print(y)

        # bepaal einduitkomst
        #s3 = np.dot(y, self.w[4,6]) + self.b[2]
        s3 = y1 * self.w[4] + y2 * self.w[5] + self.b[2]
        #print(s3)
        y3 = self.sigma(s3)
        #print(y3)
        return y3

# n = 2, dus w heeft lengte 6, b heeft lengte 3
n = len(X[0])
m = len(X)
k = 3
w = np.random.random(n*k) # 6 willekeurige getallen tussen 0 en 1
b = np.random.random(k)
p_eind = np.random.random(k+1)

netwerk = MultilayerNetwork(k, n, m, w, b, p_eind, 2, 1)




print("\nTest functies:")
for row in X:
    #uit = netwerk.output(row)
    print("In:", row, " Uit:", row )# eigenlijk uit)

netwerk.train(X, Y)

