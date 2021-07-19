__author__ = 'westbrick'
# class for multivariate randomized response

import numpy as np
import numpy.random as r


class MRR:
    name = 'MRR'

    def randomizer(self, secret):
        ep = secret[0]
        d = len(secret)-1

        exep = np.exp(ep)
        trate = exep/(exep+d-1)
        frate = 1.0/(exep+d-1)

        ipub = np.argmax(secret[1:])
        p = r.random(1)
        if p > trate-frate:
            ipub = r.choice(d, 1)

        pub = np.zeros(d, dtype=float)
        pub[ipub] = 1.0
        return np.array([exep]+list(pub))


    def decoder(self, pubs, prior="uniform", aggregation="weighted"):
        # weighted aggregation but without projecting to simplex, pubs is an array of size n*3
        d = np.size(pubs,1) - 1
        n = np.size(pubs,0)

        # debias every estimator
        trate = pubs[:,0]/(pubs[:,0]+d-1)
        frate = 1.0/(pubs[:,0]+d-1)

        for j in range(1,d+1):
            pubs[:,j] = (pubs[:,j]-frate)/(trate-frate)

        # compute relative weights with prior=uniform distribution, or unweighted aggregation results
        pf = np.array([1/d]*d)
        if prior == "unweighted":
            pf = pubs[:, 1:d+1].sum(axis=0)/n
            pf = np.clip(pf, 0.0, 1.0)
            pf = pf/pf.sum()
        #print("prior distribution", pf)
        gini = np.sum(pf*(1-pf))

        weights = 1/(gini+2/np.power(pubs[:,0]+1.0/pubs[:,0]-2, 2))
        if aggregation == "unweighted":
            weights = np.full(np.shape(pubs[:,0]), 1, dtype=float)

        #print("weights", weights)
        # compute weighted sum
        for j in range(1,d+1):
            pubs[:, j] = pubs[:, j]*weights
        fs = pubs[:, 1:d+1].sum(axis=0)/weights.sum()

        #print(fs)
        return fs











