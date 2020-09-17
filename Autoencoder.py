import torch
torch.set_num_threads(4)
from game import gametable, TableType, RandomGraphSpecs, RandomTreeSpecs
from Thor import thor, EnvSpecs, EnvType
from Odin import odin
import networkx as nx
import matplotlib.pyplot as plt
from support import baseline
import random
import time
from model import Model

def draw_the_graph(edges, nnodes):
    G = nx.DiGraph()
    ned = len(edges)
    nodes = [i for i in range(nnodes)]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    nx.draw(G)
    plt.show()


dtype = torch.float
device = torch.device("cpu")
seed = 10
random.seed(a=seed)

# define environment (maybe with external defintion files)

#Generate random graph
nnodesmin = 3
nnodesmax = 10
population_size = 100
nedges = 1e10
repetitions = 100
ndim = 30

edges = []
costs = []

nnodes = 0
for i in range(round(repetitions/population_size)):
    nnodes = nnodesmax -1

    for pop in range(population_size):
        graphspecs = {
            RandomGraphSpecs.Nnodes: nnodes,
            RandomGraphSpecs.Nedges: nedges,
            RandomGraphSpecs.Probability: 0.5,
            RandomGraphSpecs.Seed: seed + pop + i*population_size,
            RandomGraphSpecs.Repetitions: 1,
            RandomGraphSpecs.Distribution: random.gauss,  # random.uniform,
            # RandomGraphSpecs.DistParams: {"a": 1, "b": 30}
            RandomGraphSpecs.DistParams: {"mu": 10, "sigma": 4}
        }

        locedges, loccosts = gametable.table(TableType.random_graph, graphspecs)  ##TableType.random_tree, graphspecs)
        if len(locedges) > ndim:
            ndim = len(locedges)
        edges.append(locedges)
        costs.append(loccosts[0])

#draw_the_graph(edges, nnodes)
nedges = len(edges)

def embeddings(edges, costs, ndim):
    v = []
    for rep in range(len(edges)):
        vv = []
        for (i, j) in edges[rep]:
            vv.append(i + 0.0)
            vv.append(j + 0.0)
            vv.append(costs[rep][(i, j)])
        vv = vv + [0.0 for _ in range(len(vv), ndim * 3)]
        v.append(vv)
    return v


print(ndim*3)

D_in = ndim*3
D_out = D_in
specs = [
#     ("relu",nedges*4),
("relu",ndim),
#("relu",ndim*3),
#("relu",ndim*2),
    ("relu",ndim),
    ("relu", round(ndim/5)),
    ("relu",ndim),
#("relu",ndim*2),
#("relu",ndim*3),
    ("approximator",ndim),
 #("relu",ndim*3),
# ("relu",nedges*4),
       ("linear", D_out)
]
criterion = "mse"
optimizer = "adam"#
optspecs = { "lr" : 1e-5}#, "momentum": 0.1, "nesterov": True }

autoencoder = Model(D_in, specs)
autoencoder.set_loss(criterion)
autoencoder.set_optimizer(optimizer, optspecs)

nepochs = 2000

v = embeddings(edges, costs,  ndim)

batchsize = 15

for _ in range(nepochs):
    indices = random.sample(range(repetitions), repetitions)
    xx = [v[i] for i in indices]
    for b in range(round(repetitions/batchsize) - 1):
        xxt = torch.as_tensor(xx[b * batchsize : (b + 1) * batchsize])
        y_pred = autoencoder.coremdl(xxt)
        loss = autoencoder.criterion(y_pred, xxt)
        autoencoder.optimizer.zero_grad()
        loss.backward()
        autoencoder.optimizer.step()
        if b%10==0:
            print("B", b*batchsize, "loss", loss.item())


for i in range(0, repetitions, batchsize*10 ):
    # v0 = autoencoder.coremdl.layers[0](torch.as_tensor(v[i]))
    # v1 = autoencoder.coremdl.layers[1](v0)
    # v2 = autoencoder.coremdl.layers[2](v1)
    # v3 = autoencoder.coremdl.layers[3](v2)
    # v4 = autoencoder.coremdl.layers[4](v3)
    # print("0   ",v0.tolist())
    # print("1   ", v1.tolist())
    # print("2   ", v2.tolist())
    # print("3   ", v3.tolist())
    # print("4   ", v4.tolist())
    print(torch.as_tensor([v[i]]).tolist())
    print(autoencoder.coremdl(torch.as_tensor([v[i]])).tolist())

xxF = torch.as_tensor(v)
y_pred = autoencoder.coremdl(xxF)
lossF = autoencoder.criterion(y_pred, xxF)
print("F", lossF.item())

