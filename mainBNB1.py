import torch
from game import gametable, TableType, RandomGraphSpecs, RandomTreeSpecs
from BasicAlgo import basicalgo
import networkx as nx
import matplotlib.pyplot as plt
from support import baseline
import random
from environment import EnvSpecs, EnvType
import time
from randomness import randomness, ExplorationSensitivity

def draw_the_graph(edges, nnodes):
    G = nx.DiGraph()
    ned = len(edges)
    nodes = [i for i in range(nnodes)]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    nx.draw_circular(G)
    plt.show()

def test(algo, testcosts, nnodes, basevals, basetimes):
    stats = algo.test(testcosts, maximumiter=nnodes)

    final_objectives = [stat["final_objective"] for stat in stats]
    times = [stat["time"] for stat in stats]
    vy = [-f / g for f, g in zip(final_objectives, basevals)]
    vx = [f / (g + 1e-8) for f, g in zip(times, basetimes)]
    return vx, vy

dtype = torch.float
device = torch.device("cpu")
seed = 1000
random.seed(a=seed)

#Generate random graph
nnodes = 12
nedges = 1e10
repetitions = 100

graphspecs = {
    RandomGraphSpecs.Nnodes : nnodes,
    RandomGraphSpecs.Nedges : nedges,
    RandomGraphSpecs.Probability: 0.8,
    RandomGraphSpecs.Seed: seed,
    RandomGraphSpecs.Repetitions: repetitions*3,
    RandomGraphSpecs.Distribution: random.gauss, #random.uniform,#
    #RandomGraphSpecs.DistParams: {"a": 1, "b": 30}
    RandomGraphSpecs.DistParams: {"mu": 10, "sigma": 4}
}

edges, costs = gametable.table( TableType.random_graph, graphspecs)##TableType.random_tree, graphspecs)
testcosts = costs[repetitions: repetitions*3]
costs = costs[0:repetitions]
draw_the_graph(edges, nnodes)
nedges = len(edges)

basevals = []
basesols = []
for rep in range(repetitions):
    baselin = baseline()
    baseval, basesol, faketime = baselin.min_path(nnodes, edges, costs[rep])
    print("    BASELINE", basesol, " BASELINE", -baseval)
    basevals.append(baseval)
    basesols.append(basesol)

# define the structure of the network
D_in = nedges + nedges + nedges + nedges
D_out = 1
specs = [
    ("relu",nedges*8),
("relu",nedges*4),
    #("relu", nedges*2),
    #("relu", nnodes),
    #("relu", nnodes),
    #("relu", nnodes),
    #("relu", nnodes),
    ("relu", nedges),
    ("linear", D_out)
]
criterion = "mse"
optimizer = "adam"
optspecs = { "lr" : 1e-4}#, "momentum": 0.1, "nesterov": False }
scheduler = "multiplicative"
schedspecs = { "factor" : 0.999 }

# lauch the algorithm with many repetitions
memorylength = 10000
nepisodes = 20000
memorypath = None
stop_function = None

environment_specs = {
    EnvSpecs.type : EnvType.bnb1,
    EnvSpecs.statusdimension : D_in,
    EnvSpecs.actiondimension : nedges,
    EnvSpecs.rewardimension : D_out,
    EnvSpecs.prize : 1000,
    EnvSpecs.penalty : -300,
    EnvSpecs.As : As,
    EnvSpecs.bs : bs,
    EnvSpecs.cs : cs,
    EnvSpecs.N : nnodes
}

balgo = basicalgo(environment_specs=environment_specs,
                 D_in=D_in,
                 modelspecs = specs,
                 criterion=criterion,
                 optimizer=optimizer,
                 optspecs=optspecs,
                 scheduler=scheduler,
                 schedspecs=schedspecs,
                 memorylength=memorylength,
                 memorypath=memorypath,
                 seed=seed,
                 stop_function=stop_function)

stats = balgo.solve(repetitions= repetitions,
               nepisodes = nepisodes,
               display = (True, 10, True),
               randomness = randomness(r0=1, rule=ExplorationSensitivity.linear_threshold, threshold=0.02, sensitivity=0.999),
               batchsize = 15,
               maximumiter = nnodes,
               steps = 1,
               backcopy=30)



#print(stats)
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

odx, ody = test(balgo, testcosts, nnodes, basevals, basetimes)
plt.scatter(x=odx, y=ody)
plt.show()
plt.cla()