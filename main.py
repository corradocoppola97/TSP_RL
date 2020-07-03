import torch
from game import gametable, TableType, RandomGraphSpecs, RandomTreeSpecs
from Thor import thor, EnvSpecs, EnvType
from Odin import odin
import networkx as nx
import matplotlib.pyplot as plt
from support import baseline
import random
import time

def draw_the_graph(edges, nnodes):
    G = nx.DiGraph()
    ned = len(edges)
    nodes = [i for i in range(nnodes)]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    nx.draw_circular(G)
    plt.show()

def test(thoralgo, testcosts, nnodes, basevals, basetimes):
    stats = thoralgo.test(testcosts, maximumiter=nnodes)

    final_objectives = [stat["final_objective"] for stat in stats]
    times = [stat["time"] for stat in stats]
    vy = [-f / g for f, g in zip(final_objectives, basevals)]
    vx = [f / (g + 1e-8) for f, g in zip(times, basetimes)]
    return vx, vy


torch.set_num_threads(1)

dtype = torch.float
device = torch.device("cpu")
seed = 1000
random.seed(a=seed)

# define environment (maybe with external defintion files)

#Generate random graph
nnodes = 10
nedges = 1e10
repetitions = 1000

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
    ("relu",nedges*10),
    ("relu", nedges * 10),
    ("relu", nedges * 10),
# ("sigmoid",nedges*5),
# ("sigmoid",nedges*5),
# ("sigmoid",nedges*5),
# ("sigmoid",nedges*5),
#("relu",nnodes),
#("relu",nnodes),
   # ("softmax",nnodes),
     #("relu",round(nedges/3)),
   #("relu",nedges),
#("relu",nedges),
#("softmax",nnodes),
    #("relu",nnodes),
    #("logsigmoid", nedges),
    ("linear", D_out)
]
criterion = "mse"
optimizer = "adam"
optspecs = { "lr" : 1e-5}#, "momentum": 0, "nesterov": False }

# collect and plot data


# lauch the algorithm with many repetitions
memorylength = 30000
nepisodes = 10000
memorypath = None
stop_function = None

environment_specs = {
    EnvSpecs.type : EnvType.minpath,
    EnvSpecs.statusdimension : D_in,
    EnvSpecs.actiondimension : nedges,
    EnvSpecs.rewardimension : D_out,
    EnvSpecs.edges : edges.copy(),
    EnvSpecs.costs : costs.copy(),
    EnvSpecs.prize : 0,
    EnvSpecs.penalty : -10000,
    EnvSpecs.finalpoint : nnodes-1,
    EnvSpecs.startingpoint : 0
}

odinalgo = odin(environment_specs,
                 D_in,
                 specs,
                 criterion,
                 optimizer,
                 optspecs,
                 memorylength,
                 memorypath,
                 seed,
                 stop_function,
                 repetitions)
#
# stats = odinalgo.pretrain(nepisodes = nepisodes,
#                display = (True, 10),
#                randomness0 = 1,
#                batchsize = 15,
#                maximumiter = nnodes,
#                steps = 1
#                )
stats = odinalgo.solve(nepisodes = nepisodes,
               display = (True, 10),
               randomness0 = 1,
               batchsize = 15,
               maximumiter = nnodes,
               steps = 1
               )



#print(stats)
final_objectives = [stat["final_objective"] if stat["is_final"] == 1 else -200 for stat in stats ]
plotbasevals = [[-baseval for i in range(len( final_objectives))] for baseval in basevals ]


for plotbase in plotbasevals:
    plt.plot(plotbase)
plt.plot(final_objectives)
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

odx, ody = test(odinalgo, testcosts, nnodes, basevals, basetimes)



thoralgo = thor(environment_specs,
                 D_in,
                 specs,
                 criterion,
                 optimizer,
                 optspecs,
                 memorylength,
                 memorypath,
                 seed,
                 stop_function,
                 repetitions)

stats = thoralgo.solve(nepisodes = nepisodes,
               display = (True, 10),
               randomness0 = 1,
               batchsize = 15,
               maximumiter = nnodes,
               steps = 1
               )



#print(stats)
final_objectives = [stat["final_objective"] if stat["is_final"] == 1 else -200 for stat in stats ]
plotbasevals = [[-baseval for i in range(len( final_objectives))] for baseval in basevals ]


for plotbase in plotbasevals:
    plt.plot(plotbase)
plt.plot(final_objectives)
plt.show()
plt.cla()

gaps = [abs((stat["final_objective"]+basevals[stat["rep"]])/basevals[stat["rep"]])   for stat in stats if stat["is_final"] == 1]
bottom = [0 for _ in  range(len(gaps))]

plt.plot(gaps)
plt.plot(bottom)
plt.show()
plt.cla()


thx, thy = test(odinalgo, testcosts, nnodes, basevals, basetimes)
plt.scatter(x=odx, y=ody)
plt.scatter(x=thx, y=thy)
plt.show()
plt.cla()


# cumulatives = [stat["cumulative_reward"] for stat in stats if stat["is_final"] == 1]
# maxlen = 0
#
# for i in range(0,len(cumulatives), math.ceil(nepisodes/20)):
#     plt.plot(cumulatives[i])
#     if len(cumulatives[i]) > maxlen:
#         maxlen = len(cumulatives[i])
# basevals = [-baseval for i in range(maxlen)]
# plt.plot(basevals)
# plt.show()
# bestsol = -1e10
# bestsolpath = []
# bestsol2 = bestsol
# bestsolpath2 = []
# for stat in stats:
#     if -baseval + 1 >= stat["final_objective"] >= bestsol:
#         bestsol = stat["final_objective"]
#         bestsolpath = stat["solution"]
#     if  stat["final_objective"] >= bestsol2:
#         bestsol2 = stat["final_objective"]
#         bestsolpath2 = stat["solution"]
# print("    BASELINE", basesol,      " BASELINE", -baseval)
# print("BESTSOL PATH", bestsolpath,  "  BESTSOL", bestsol)
# print("BESTSOL2PATH", bestsolpath2, " BESTSOL2", bestsol2)
# print("LASTSOL2PATH", stats[len(stats)-1]["solution"], " LASTSOL2", stats[len(stats)-1]["final_objective"],"  Is final?", True if stats[len(stats)-1]["is_final"] == 1 else False)
# print("edges",edges)
# print("costs",costs)













































"""
D_in = nedges + nedges + nedges + nedges
D_out = 1
# specs = [
#     ("relu", nedges),("relu", nedges),("relu", nedges),
#     #("relu", nnodes),
#     #("logsigmoid", nedges),
#     ("linear", D_out)
# ]
# print(specs)
criterion = "mse"
optimizer = "sgd"
optspecs = { "lr" : 1e-6, "momentum": 0, "nesterov":False }


# lauch the algorithm with many repetitions
memorylength = 500000
memorypath = None
stop_function = None

environment_specs = {
    EnvSpecs.type : EnvType.minpath,
    EnvSpecs.statusdimension : D_in,
    EnvSpecs.actiondimension : nedges,
    EnvSpecs.rewardimension : D_out,
    EnvSpecs.edges : edges,
    EnvSpecs.costs : costs,
    EnvSpecs.prize : 0,
    EnvSpecs.penalty : -1000,
    EnvSpecs.finalpoint : nnodes-1,
    EnvSpecs.startingpoint : 0
}

thoralgo = thor(environment_specs,
                 D_in,
                 specs,
                 criterion,
                 optimizer,
                 optspecs,
                 memorylength,
                 memorypath,
                 seed,
                 stop_function,
                 repetitions = repetitions)

nepisodes = 2000
stats = thoralgo.solve(nepisodes = nepisodes,
               display = (True, 10),
               randomness0 = 1,
               batchsize = 5,
               maximumiter = nnodes,
               steps = 1
               )


final_objectives = [stat["final_objective"] if stat["is_final"] == 1 else -200 for stat in stats ]
plotbasevals = [[-baseval for i in range(len( final_objectives))] for baseval in basevals ]


for plotbase in plotbasevals:
    plt.plot(plotbase)
plt.plot(final_objectives)
plt.show()
plt.cla()

gaps = [abs((stat["final_objective"]+basevals[stat["rep"]])/basevals[stat["rep"]])   for stat in stats if stat["is_final"] == 1]

plt.plot(gaps)
plt.show()
plt.cla()
"""


