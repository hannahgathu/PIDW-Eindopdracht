import numpy as np

class NeuralNetwork():

    def __init__(self, X, Y):
        """initialiseert een neuraal netwerk:           
           X is de invoer van data om te leren,
           Y is de bekende uitvoer bij de testdata
           """
        w_len = np.shape(X)[1]
        #p is een list met beginschattingen voor de gewichten [w1,w2,...wn,b]
        self.p = np.random.random(w_len + 1)
        self.X = X
        self.Y = Y

    def sigma(self, x):
        """sigma (activatiefunctie)"""
        return 1/(1+np.exp(-x))

    def gradient(self, X, Y):
        """berekent de gradient van de errorfunctie"""
        p = self.p
        #s is voor ieder voorbeeld het inproduct van X en W
        s = np.dot(X,p[:-1]) + p[-1]
        yt = self.sigma(s)
        g = np.zeros(len(p))
        #len(yt) = aantal voorbeelden
        g[:-1] = 2*np.dot(np.transpose(X)*yt*(1 - yt)*(yt - Y),np.ones(len(yt)))
        g[-1]  = 2*np.dot(yt*(1 - yt)*(yt - Y),np.ones(len(yt)))
        return g
            
    def train(self, iteraties, alfa):
        """X is de invoer, y de uitvoer"""
        for iteration in range(iteraties):
            g = self.gradient(self.X, self.Y)
            self.p = self.p-alfa*g

    def predict(self, inputs):
        """berekent de uitvoer voor de gegeven 'inputs'
           adhv de gevonden waarden van p
           """
        return self.sigma(np.dot(inputs, self.p[:-1])+self.p[-1])
