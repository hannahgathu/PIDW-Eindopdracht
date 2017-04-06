    def bereken_y(self, invoer, p):
        """ bereken de uitvoer van het netwerk """
        if len(np.shape(invoer)) is 1: # this makes a two-dim matrix from a list
            invoer = np.array([invoer])
        w = p[:-(2*self.k+1)] # neem alle gewichten van input naar laag
        w = np.reshape(w, [np.shape(invoer)[-1], -1], 'F') # maak een matrix: rij<>l ; kol<>k
        d = np.dot(invoer, w) # matrix product invoer en w: rij<>j ; kol<>k
        b = np.tile(p[-(self.k+1):-1], (np.shape(d)[0],1)) # maak array met bias vlnr
        y = self.sigma(b + d) # bepaal y = sigma(<x,w> + b) voor alle k en j
        yw = y * self.p[-(2*self.k+1):-(self.k+1)] # bepaal y * w`
        s_uit = p[-1] + np.sum(yw, axis=1) # sommeer rijen en voeg bias_out toe
        return self.sigma(s_uit)
