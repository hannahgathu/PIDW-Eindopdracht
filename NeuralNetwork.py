import numpy as np

class NeuralNetwork():

    def __init__(self, w, iterations, alpha):
        """initialiseert een neuraal netwerk:
           w is een list met beginschattingen voor de gewichten
           [w1,w2,...wn,b]"""
        self.p = np.array(w)
        self.iterations = iterations
        self.alpha = alpha

    def sigma(self, x):
        """sigma (activatiefunctie)"""
        return 1/(1+np.exp(-x))

    def gradient(self, X, y):
        """berekent de gradient van de errorfunctie"""
        p = self.p
        #s is voor ieder voorbeeld het inproduct van X en W
        s = np.dot(X,p[:-1]) + p[-1]
        yt = self.sigma(s)
        g = np.zeros(len(p))
        #len(yt) = aantal voorbeelden
        g[:-1] = 2*np.dot(np.transpose(X)*yt*(1 - yt)*(yt - y),np.ones(len(yt)))
        g[-1]  = 2*np.dot(yt*(1 - yt)*(yt - y),np.ones(len(yt)))
        return g
            
    def train(self, X, y):
        """X is de invoer, y de uitvoer"""
        for iteration in range(self.iterations):
            g = self.gradient(X, y)
            self.p = self.p-self.alpha*g

    def predict(self, inputs):
        """berekent de uitvoer voor de gegeven 'inputs'
           adhv de gevonden waarden van p"""
        return self.sigma(np.dot(inputs, self.p[:-1])+self.p[-1])
