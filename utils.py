__author__ = 'westbrick'
# util functions

import math
import numpy as np
import scipy as sp
import numpy.random as r
import mrr
import utils
import decimal
from decimal import Decimal as D
import time

def binarysearch(l, v):
    # search the corresponding scope holding v
    s = 0
    e = len(l)-1
    while v < l[math.floor((s+e)/2)] or v >= l[math.floor((s+e)/2) + 1]:
        if v < l[math.floor((s+e)/2)]:
            e = math.floor((s+e)/2)
        else:
            s = math.floor((s+e)/2)
    return math.floor((s+e)/2)

def bitarrayToList(ba):
    return [i for i in range(0, len(ba)) if ba[i] > 0]

def reservoirsample(l, m):
    # sample m elements from list l
    samples = l[0:m]
    for i in range(m, len(l)):
        index = r.randint(0, i+1)
        if index < m:
            samples[index] = l[i]
    return samples


def recorder(hits, pub):
    # record pub to hits
    hits += pub
    return hits

def distributor(n, datas, mechanism, prior, aggregation):
    # randomize data and return estimated distribution
    pubs = np.full(np.shape(datas), 0.0, dtype=float)
    for i in range(0, n):
        pubs[i] = mechanism.randomizer(datas[i])

    return mechanism.decoder(pubs, prior, aggregation)

def projector(od):
    # project od to probability simplex
    u = -np.sort(-od)
    #print("sorted:\t", u)
    sod = np.zeros(len(od))
    sod[0] = u[0]
    for i in range(1, len(od)):
        sod[i] = sod[i-1]+u[i]

    for i in range(0, len(od)):
        sod[i] = u[i]+(1.0-sod[i])/(i+1)

    p = 0
    for i in range(len(od)-1, -1, -1):
        if sod[i] > 0.0:
            p = i
            break

    q = sod[p]-u[p]

    x = np.zeros(len(od))
    for i in range(0, len(od)):
        x[i] = np.max([od[i]+q, 0.0])
    #print("projected:\t",x)
    return x



def randomDatas(n, d, m, dist=None, epdist=None, online=False, eplist=None):
    # ensure +1-1 pairs only have one non-zero entry
    datas = np.zeros((n, 1+d), dtype=float)

    if dist is None:
        dist = np.ones(d, dtype=float)
    else:
        dist = np.array(dist[0:d])
    dist = dist/dist.sum()
    dist = dist*n
    if dist.sum() != n:
        dist[-1] += n-dist.sum()

    if epdist is None:
        epdist = np.ones(len(eplist), dtype=float)
    else:
        epdist = np.array(epdist)
    epdist = epdist/epdist.sum()



    for i in range(0, n):
        # get (random) privacy budget
        p = r.random(1)
        pi = -1
        psum = 0.0
        while psum <= p:
            pi += 1
            psum += epdist[pi]
        ep = eplist[pi]
        if online:
            # compute online local budget
            if i > 0:
                delta = 1/i
                clones = (1/np.exp(datas[0:i,0])).sum()
                if clones >= 16*np.log(2*i):
                    fep = (np.exp(ep)-1.0)*clones/(8*(1+np.sqrt(clones*np.log(4*i))))
                    lexep = (fep+1)/(1-fep)
                    if lexep >= np.exp(ep):
                        ep = np.log(lexep)
        # get (pesudo-random) data value
        rdi = 1
        if n-i > 1:
            rdi = r.randint(1, n-i)
        di = -1
        isum = 0
        #print("rdi, dist", i, rdi, dist)
        while isum+0.01 < rdi:
            di += 1
            isum += dist[di]
        dist[di] -= 1
        data = np.zeros(d, dtype=float)
        data[di] = 1.0
        datas[i] = np.array([ep]+list(data))

    #print(datas)
    return datas


def initmechanisms(mechanisms, d, m, ep):
    # instance mechanisms with concrete setting
    mechanism_instances = []
    for mk in mechanisms:
        mechanism_instances.append(mrr.MRR())

    return mechanism_instances


def countsToSizes(dl):
    # compute m-size:probability pairs from estimated padding items
    sizes = np.full(len(dl)+1, 0, dtype=float)
    sizes[0] = 1.0 - dl[0]
    for i in range(0, len(dl)-1):
        sizes[i+1] = dl[i] - dl[i+1]
    sizes[len(dl)] = dl[-1]
    #print(sizes.tolist())
    return sizes


def Comb(n, k):
    b = D(1.0)
    for i in range(0, k):
        b = b*D((n-i)/(i+1))
    return b

__name__=["Comb"]
