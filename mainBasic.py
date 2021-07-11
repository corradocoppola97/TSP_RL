import torch
import torch.nn as nn
from Game_GH import TableType, RandomGraphSpecs, gametable
from BasicAlgo import basicalgo
#import networkx as nx
import matplotlib.pyplot as plt
#from support import baseline
import random
from Environment_GH import EnvSpecs, EnvType, tsp
import time
from randomness import randomness, ExplorationSensitivity
from modules2 import *
from TSP_ortools import risolvi_problema, print_solution
from support import baseline_ortools

# def draw_the_graph(edges, nnodes):
#     G = nx.DiGraph()
#     ned = len(edges)
#     nodes = [i for i in range(nnodes)]
#     G.add_nodes_from(nodes)
#     G.add_edges_from(edges)
#
#
#     figure = plt.figure(figsize=(10,10))
#     nx.draw_circular(G)
#     plt.show()

def test(algo, testcosts, nnodes, basevals, basetimes):
    stats = algo.test(testcosts, maximumiter=nnodes)

    final_objectives = [stat["final_objective"] for stat in stats]
    times = [stat["time"] for stat in stats]
    vy = [-f / g for f, g in zip(final_objectives, basevals)]
    vx = [f / (g + 1e-8) for f, g in zip(times, basetimes)]
    return vx, vy

dtype = torch.float
device = torch.device("cuda:0")
seed = 20000
random.seed(a=seed)

#Generate random graph
nnodes = 10
nedges = nnodes*(nnodes-1)
repetitions = 10

graphspecs = {
    RandomGraphSpecs.Nnodes : nnodes,
    RandomGraphSpecs.Nedges : nedges,
    RandomGraphSpecs.Probability: None,
    RandomGraphSpecs.Seed: seed,
    RandomGraphSpecs.Repetitions: repetitions*3,
    RandomGraphSpecs.Distribution: None,
    RandomGraphSpecs.DistParams: (0,100,2)
}

# depth = 5
# nnodes = pow(2,depth)
# repetitions = 100
#
# graphspecs = {
#     RandomTreeSpecs.Depth : depth,
#     RandomTreeSpecs.Seed: seed,
#     RandomTreeSpecs.Repetitions: repetitions,
#     RandomTreeSpecs.Distribution: random.uniform,#random.gauss,
#     RandomTreeSpecs.DistParams: {"a": 1, "b": 30}
#     #RandomGraphSpecs.DistParams: {"mu": 10, "sigma": 4}
# }

edges, costs1, ds = gametable._random_graph_distances(graphspecs)
testcosts = costs1[repetitions: repetitions*3]
costs = costs1[0:repetitions]
#nedges = len(edges)
'''
basevals = []
basesols = []
for rep in range(repetitions):
    baselin = baseline()
    baseval, basesol, faketime = baselin.min_path(nnodes, edges, costs[rep])
    print("    BASELINE", basesol, " BASELINE", -baseval)
    basevals.append(baseval)
    basesols.append(basesol)
'''

# define the structure of the network
D_in = 4*nnodes + nedges + 1
D_out = 1
specs = [("relu",D_in),("relu",150),("sigmoid", 75),("sigmoid", 30),('relu', 10),('linear',1)]
criterion = "mse"
optimizer = "adam"
optspecs = { "lr" : 1e-3}#, "momentum": 0.1, "nesterov": False }
scheduler = None
schedspecs = {"factor":1.0}




# lauch the algorithm with many repetitions
memorylength = 10000
nepisodes = 5000
memorypath = None
stop_function = None

environment_specs = {
    EnvSpecs.type : EnvType.tsp,
    EnvSpecs.statusdimension : D_in,
    EnvSpecs.actiondimension : nedges,
    EnvSpecs.rewardimension : D_out,
    EnvSpecs.edges : edges.copy(),
    EnvSpecs.costs : costs1.copy(),
    EnvSpecs.prize : 1000,
    EnvSpecs.penalty : 0,
    EnvSpecs.finalpoint : nnodes-1,
    EnvSpecs.startingpoint : 0
}

mod_layers = {}
mod_layers['CNNst'] = nn.Sequential(nn.Conv2d(1,8,3,stride=1,padding=1),nn.ReLU(),
                                    nn.Conv2d(8,16,3,stride=1,padding=1),nn.ReLU(),
                                    nn.Conv2d(16,64,3,stride=1,padding=1))
mod_layers['CNNst1'] = mod_layers['CNNst']
mod_layers['MLP'] = nn.Sequential(nn.Linear(nnodes+1,1),nn.ReLU(),nn.Linear(1,nnodes+1))

mod_layers['encoder'] = nn.Sequential(nn.Linear(nnodes+1+int(((nnodes**2)+(nnodes+1)**2)),75),nn.ReLU(),
                                    nn.Linear(75,30),nn.ReLU(),
                                    nn.Linear(30,10),nn.ReLU(), nn.Linear(10,1))

#mod_layers = nn.Sequential(nn.Linear(nnodes+1+int(nnodes**2 + (nnodes+1)**2),110),nn.ReLU(),
                        #nn.Linear(110,50),nn.ReLU(),nn.Linear(50,10),nn.ReLU(),nn.Linear(10,1))

balgo = basicalgo(environment_specs=environment_specs,

                 D_in= D_in,
                 modelspecs = specs,
                 criterion=criterion,
                 optimizer=optimizer,
                 optspecs=optspecs,
                 scheduler=scheduler,
                 schedspecs=schedspecs,
                 memorylength=memorylength, mod_layers=mod_layers,
                 memorypath=memorypath,
                 seed=seed,
                 stop_function=stop_function, nnodes = nnodes, edges = edges)

stats = balgo.solve(repetitions= repetitions,
               nepisodes = nepisodes,
               noptsteps = 1,
               display = (False,0),
               randomness = randomness(r0=1, rule=ExplorationSensitivity.linear_threshold, threshold=0.01, sensitivity=0.999),
               batchsize = 1,
               maximumiter = nnodes,
               steps = 1,
               backcopy=0)




#print(stats)
import matplotlib.pyplot as plt
import numpy as np
def print_stats(stats,m):
    N = len(stats)
    mov_avg = []
    curr = [stats[i]['final_objective'] for i in range(0,m)]
    for i in range(m,N):
        mmob = sum(curr)/m
        mov_avg.append(mmob)
        rew_i = stats[i]['final_objective']
        curr.pop(0)
        curr.append(rew_i)

    ff = len(mov_avg)
    num = np.arange(0,ff)
    mavg = np.array(mov_avg)
    return num,mavg

def ortools_solutions(mat):
    l = []
    for matrix in mat:
        d = {}
        d['distance_matrix'] = matrix
        d['num_vehicles'] = 1
        d['depot'] = 0
        n = len(matrix)
        sol, man, rout = risolvi_problema(n,d)
        l.append(sol.ObjectiveValue())
    return l

test_stats = balgo.test(testcosts)
aaa = baseline_ortools(dis_m=ds)
ortools_values,comp_time,outputs = aaa.solve_ortools(printFLAG=True)
tr,vr = aaa.TestPlot(ortools_values,comp_time,test_stats)





'''
final_objectives = [stat["final_objective"] if stat["is_final"] == 1 else 0 for stat in stats ]
plotbasevals = [[-baseval for i in range(len( final_objectives))] for baseval in basevals ]


for plotbase in plotbasevals:  
    plt.plot(plotbase)
plt.show()
plt.cla()
for plotbase in plotbasevals:
    plt.plot(plotbase)
plt.plot(final_objectives,'o')
plt.show()
plt.cla()

gaps = [abs((stat["final_objective"]+basevals[stat["rep"]])/basevals[stat["rep"]])   for stat in stats if stat["is_final"] == 1]
bottom = [0 for _ in  range(len(gaps))]

plt.plot(gaps)
plt.plot(bottom)
plt.show()
plt.cla()

nedges = len(edges)
basevals = []
basesols = []
basetimes = []

for rep in range(repetitions):
    baselin = baseline()
    baseval, basesol, basetime = baselin.min_path(nnodes, edges, testcosts[rep])
    basevals.append(baseval)
    basesols.append(basesol)
    basetimes.append(basetime)
    # basevals.append(1)
    # basesols.append([0,14])
    # basetimes.append(1)

odx, ody = test(balgo, testcosts, nnodes, basevals, basetimes)
plt.scatter(x=odx, y=ody)
plt.show()
plt.cla()
'''