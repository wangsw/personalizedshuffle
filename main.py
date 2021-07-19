__author__ = 'westbrick'

import simulator
import numpy as np

#n = 10000
n = 48842
#n = 569
# ds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
# ds = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# ds = [12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
#ds = [8]
#ms = [1]
#ds = [2, 4, 8, 16]
ds = [2]
#ds = [1024-16, 8192-16]
ms = [1]
#ms = [4,8,16,32]
#ms = [8, 10, 20, 30]
# ds = [40, 50]
# eps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0]
# eps = [0.001, 0.01, 0.05, 0.1, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0,  5.0]
#eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
#eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
#eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
#eps = [0.01, 0.05, 0.1, 0.2, 0.4]
#eps = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
#eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
eps = [1.0]
#eps = [0.001, 0.01, 0.1, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

repeat = 400
#pss = [np.arange(nds[i]) for i in range(0, len(nds))]
#pss = [np.sort(np.random.choice(np.arange(nds[i]*2), nds[i], False)) for i in range(0, len(nds))]

sets = None
#dist = [0.5, 0.5]
#dist = [0.2, 0.8]
#dist = [32650/48842, 16192/48842] # UCI-adult gender
dist = [37155/48842, 11687/48842] # UCI-adult income
#dist = [212/569, 357/569] # breast cancer

eplist = [0.01, 0.1, 0.5, 1.0, 2.0]
#eplist = [np.log(1.22212), np.log(5.47715), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.19428), np.log(5.3524), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.17334), np.log(5.25854), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.16832), np.log(5.23605), np.log(23.4664), 1.0, 2.0]

#eplist = [np.log(1.07698), np.log(2.26718), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.005), np.log(2.1634), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.005), np.log(2.05894), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.005), np.log(1.98315), np.log(8.88787), 1.0, 2.0]

#eplist = [np.log(1.02723), np.log(1.32714), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.005), np.log(1.31728), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.005), np.log(1.3082), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.005), np.log(1.28424), np.log(5.75559), 1.0, 2.0]


#eplist = [np.log(np.exp(0.01)), np.log(3.85772), np.log(17.2891), 1.0, 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(3.48542), np.log(10.2757), 1.0, 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(3.13552), np.log(12.3859), 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(3.1355), np.log(12.3859), np.log(12.3864), 2.0]

#eplist = [np.log(np.exp(0.01)), np.log(1.8419), np.log(8.25478), 1.0, 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(1.73045), np.log(7.75534), 1.0, 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(1.65344), np.log(7.40451), 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(1.62824), np.log(7.29728), np.log(32.7041), 2.0]

#eplist = [np.log(np.exp(0.01)), np.log(1.28669), np.log(5.76665), 1.0, 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(1.22659), np.log(5.49721), 1.0, 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(1.08055), np.log(4.84271), 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(1.25792), np.log(5.63762), np.log(1.005), 2.0]

#Adult
#eplist = [np.log(1.1593), np.log(np.exp(0.1)), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.13939), np.log(5.10641), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.02401), np.log(4.58931), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.005), np.log(4.41178), np.log(19.7722), 1.0, 2.0]

#eplist = [np.log(np.exp(0.01)), np.log(2.90235), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(2.62725), np.log(11.7745), 1.0, 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(2.42826), np.log(10.8827), 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(2.39664), np.log(10.741), np.log(22.7457), 2.0]

#Adult direct
eplist = [np.log(1.1593), np.log(np.exp(0.1)), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.13514), np.log(2.01639), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.11577), np.log(2.31675), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.11577), np.log(2.01639), np.log(6.19001), 1.0, 2.0]

#eplist = [np.log(np.exp(0.01)), np.log(2.90235), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(2.56998), np.log(6.53667), 1.0, 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(2.31675), np.log(7.11669), 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(2.31675), np.log(6.53667), np.log(17.1434), 2.0]

#BreastCancer
#eplist = [np.log(1.02116), np.log(np.exp(0.1)), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.02119), np.log(1.005), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.005), np.log(1.23475), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(1.005), np.log(1.005), np.log(4.33523), 1.0, 2.0]

#eplist = [np.log(np.exp(0.01)), np.log(1.22199), np.log(np.exp(0.5)), 1.0, 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(1.005), np.log(3.97305), 1.0, 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(1.005), np.log(3.39415), 2.0]
#eplist = [np.log(np.exp(0.01)), np.log(1.00502), np.log(3.23132), np.log(14.3965), 2.0]



epdist = [1.0, 0.0, 0.0, 0.0, 0.0]
#epdist = [0.7, 0.3, 0.0, 0.0, 0.0]
#epdist = [0.5, 0.5, 0.0, 0.0, 0.0]
#epdist = [0.5, 0.3, 0.2, 0.0, 0.0]

#epdist = [0.0, 1.0, 0.0, 0.0, 0.0]
#epdist = [0.0, 0.7, 0.3, 0.0, 0.0]
#epdist = [0.0, 0.5, 0.5, 0.0, 0.0]
#epdist = [0.0, 0.5, 0.3, 0.2, 0.0]
online = False
prior = "uniform"
aggregation = "weighted"

print('n=',n)
print('ds=', ds)
print('dist=', dist)
print('ms=', ms)
print('eps=', eps)
print('eplist=', eplist)
print('epdist=', epdist)
print('online, prior, aggregation=', online, prior, aggregation)
#mechanisms = ['MRR', 'BRR', 'GBFMM', 'HBFMM', 'KSS', 'EM', 'EKSE', 'OKSE', 'ECSE', 'OCSE']

mechanisms = ["MRR"]


simulation = simulator.Simulator()
simulation.init(n, ds, ms, eps, repeat, mechanisms, sets, dist, epdist, online, eplist, prior, aggregation)
simulation.simulate()
simulation.write('random_u1000')


#testing = tester.Tester()
#testing.init(n, ds, ms, eps, repeat, mechanisms, dist)
#testing.test()
#testing.write('uniform_10000')



