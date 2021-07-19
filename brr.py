__author__ = 'westbrick'
# class for binary randomized response

import numpy as np
import numpy.random as r
import time

class BRR:
    name = 'BRR'
    ep = 0.0    # privacy budget epsilon

    d = 0 # domain size + maximum subset size
    m = 0 # maximum subset size
    trate = 0.0 # hit rate when true
    frate = 0.0 # hit rate when false

    # records for consuming time
    clienttime = 0.0
    recordtime = 0.0
    servertime = 0.0

    def __init__(self, d, m, ep, k=None):
        self.ep = ep
        self.d = d
        self.m = m
        # records for consuming time
        self.clienttime = 0.0
        self.recordtime = 0.0
        self.servertime = 0.0

        self.__setparams()

    def __setparams(self):
        self.trate = np.exp(self.ep/(2*self.m))/(np.exp(self.ep/(2*self.m))+1)
        self.frate = 1.0/(np.exp(self.ep/(2*self.m))+1)
        # print(np.exp(self.ep/2), self.trate, self.frate)


    def randomizer(self, secrets):
        tstart = time.process_time()
        pub = np.zeros(self.d, dtype=int)
        for i in range(0, self.d):
            p = r.random(1)
            if secrets[i] > 0:
                if p < self.trate:
                    pub[i] = 1
                else:
                    pub[i] = 0
            else:
                if p < self.frate:
                    pub[i] = 1
                else:
                    pub[i] = 0
        self.clienttime += time.process_time()-tstart
        return pub

    def decoder(self, hits, n):
        # debias hits but without projecting to simplex
        #print('rates', self.trate, self.frate)
        tstart = time.process_time()
        fs = np.array([(hits[i]-n*self.frate)/(self.trate-self.frate) for i in range(0, self.d)])
        self.servertime += time.process_time()-tstart
        return fs/n

    def bound(self, n, tfs=None):
        # compute theoretical squared l2-norm error bound
        return (self.trate*(1.0-self.trate)+(self.d-1)*self.frate*(1-self.frate))/(n*(self.trate-self.frate)*(self.trate-self.frate))
