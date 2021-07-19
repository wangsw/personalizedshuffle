__author__ = 'westbrick'
# simulator for local private discrete distribution estimation

import json
import numpy as np
import scipy as sp
import numpy.random as r

import utils
from distance import *
from datetime import datetime, date, time

class Simulator:
    n = 0  # number of providers
    ds = None  # a list of domain size
    ms = None  # a list of maximum subset size
    eps = None  # a list of privacy budget epsilons
    sets = None  # subsets as a binary array
    results = {}  # dict to record simulation settings and results
    mechanisms = None
    dist = None  # representing the probabilities each item shows
    epdist = None # distribution of privacy budget
    repeat = 100  # repeat time for each simulation

    def init(self, n, ds, ms, eps, repeat, mechanisms, sets=None, dist=None, epdist=None, online=False, eplist=None, prior="uniform", aggregation="weighted"):
        self.n = n
        self.ds = ds
        self.ms = ms
        self.eps = eps
        self.sets = sets
        self.repeat = repeat
        self.mechanisms = mechanisms
        self.dist = dist
        self.epdist = epdist
        self.eplist = eplist
        self.online = online
        self.prior = prior
        self.aggregation = aggregation
        self.results['n'] = self.n
        self.results['ds'] = self.ds
        self.results['ms'] = self.ms
        self.results['eps'] = self.eps
        self.results['sets'] = self.sets
        self.results['repeat'] = self.repeat
        self.results['dist'] = self.dist
        self.results['mechanisms'] = self.mechanisms
        for d in ds:
            self.results['d'+str(d)] = {}
            for m in [mi for mi in self.ms if mi <= d]:
                self.results['d'+str(d)]['m' + str(m)] = {}
                for ep in eps:
                    self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)] = {}
                    self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['histograms'] = [None]*self.repeat
                    for mk in self.mechanisms:
                        self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['estimators_'+mk] = [None]*self.repeat
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_raw_mean_l2_' + mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_raw_mean_l1_' + mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_raw_mean_linf_' + mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_raw_mean_kld_' + mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['num_raw_mean_l2_' + mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['num_raw_mean_l1_' + mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['num_raw_mean_linf_' + mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['num_raw_mean_kld_' + mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_mean_l2_'+mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_mean_l1_'+mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_mean_linf_'+mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_mean_kld_' + mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['num_mean_l2_'+mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['num_mean_l1_'+mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['num_mean_linf_'+mk] = 0.0
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['num_mean_kld_' + mk] = 0.0


    def simulate(self):
        for di in range(0, len(self.ds)):
            d = self.ds[di]
            for m in [m for m in self.ms if m <= d]:
                for ep in self.eps:
                    # initialize mechanisms
                    mechanism_instances = utils.initmechanisms(self.mechanisms, d, m, ep)
                    # continue
                    print('d=', d, ', m=', m, ', epsilon=', ep, ', starts')
                    for rt in range(0, self.repeat):
                        sets = None
                        if self.sets is None:
                            sets = utils.randomDatas(self.n, d, m, self.dist, self.epdist, self.online, self.eplist)
                        else:
                            sets = self.sets
                        h = sets[:, 1:].sum(axis=0)
                        # print("sets & h", sets, h)
                        self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['histograms'][rt] = h.tolist()
                        for mk in mechanism_instances:
                            # print('mechanism', mk.name)
                            # randomizer and decoder
                            neh = utils.distributor(self.n, sets, mk, self.prior, self.aggregation)
                            self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['estimators_'+mk.name][rt] = neh.tolist()
                            nh = h/self.n
                            #neh = eh/self.n


                            #print("time", mk.name, mk.clienttime, mk.recordtime, mk.servertime, self.n, d, ep)

                            #print("h eh", h.tolist(), eh.tolist(), mk.name)

                            #print(nh, ws, nwh)
                            self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_raw_mean_l2_'+mk.name] += l2norm(nh, neh)
                            self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_raw_mean_l1_'+mk.name] += l1norm(nh, neh)
                            self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_raw_mean_linf_'+mk.name] += infnorm(nh, neh)
                            self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_raw_mean_kld_'+mk.name] += KLdivergence(nh, neh)


                            npeh = utils.projector(neh)
                            #print(nh, npeh, l2norm(nh, npeh))

                            self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_mean_l2_'+mk.name] += l2norm(nh, npeh)
                            self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_mean_l1_'+mk.name] += l1norm(nh, npeh)
                            self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_mean_linf_'+mk.name] += infnorm(nh, npeh)
                            self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_mean_kld_'+mk.name] += KLdivergence(nh, npeh)

                        # print('iteration=', rt, ' ends')


                    for mk in mechanism_instances:
                        self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_raw_mean_l2_'+mk.name] /= self.repeat
                        self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_raw_mean_l1_'+mk.name] /= self.repeat
                        self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_raw_mean_linf_'+mk.name] /= self.repeat
                        self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_raw_mean_kld_'+mk.name] /= self.repeat
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)][
                            'num_raw_mean_l2_' + mk.name] /= self.repeat
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)][
                            'num_raw_mean_l1_' + mk.name] /= self.repeat
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)][
                            'num_raw_mean_linf_' + mk.name] /= self.repeat
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)][
                            'num_raw_mean_kld_' + mk.name] /= self.repeat

                        self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_mean_l2_'+mk.name] /= self.repeat
                        self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_mean_l1_'+mk.name] /= self.repeat
                        self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_mean_linf_'+mk.name] /= self.repeat
                        self.results['d'+str(d)]['m' + str(m)]['ep'+str(ep)]['core_mean_kld_'+mk.name] /= self.repeat

                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)][
                            'num_mean_l2_' + mk.name] /= self.repeat
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)][
                            'num_mean_l1_' + mk.name] /= self.repeat
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)][
                            'num_mean_linf_' + mk.name] /= self.repeat
                        self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)][
                            'num_mean_kld_' + mk.name] /= self.repeat

                        print('core_raw_' + mk.name,
                              self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_raw_mean_l2_' + mk.name],
                              self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_raw_mean_l1_' + mk.name],
                              self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_raw_mean_linf_' + mk.name],
                              self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_raw_mean_kld_' + mk.name])



                        print('core_prj_' + mk.name,
                              self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_mean_l2_' + mk.name],
                              self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_mean_l1_' + mk.name],
                              self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_mean_linf_' + mk.name],
                              self.results['d' + str(d)]['m' + str(m)]['ep' + str(ep)]['core_mean_kld_' + mk.name])
                        # print

                    # print('d=', d, ', epsilon=', ep, ', ends')

    def write(self, filename):
        with open("logs/"+datetime.now().isoformat().replace(':', '_')+'-'+filename, 'w') as outfile:
            json.dump(self.results, outfile)

    def read(self, filename):
        with open(filename, 'r') as data_file:
            self.results = json.load(data_file)
        self.n = self.results['n']
        self.ds = self.results['ds']
        self.ms = np.array(self.results['ms'])
        self.eps = self.results['eps']
        self.repeat = self.results['repeat']













