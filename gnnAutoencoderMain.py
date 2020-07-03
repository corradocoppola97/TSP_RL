import torch
from game import gametable, TableType, RandomGraphSpecs, RandomTreeSpecs
from Thor import thor, EnvSpecs, EnvType
from Odin import odin
import networkx as nx
import matplotlib.pyplot as plt
from support import baseline
import random
import time
from model import Model
from sys import float_info as fi

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

nedges = 1e10
repetitions = 1000
realrepet = repetitions*3

nnodes = 100

graphspecs = {
    RandomGraphSpecs.Nnodes: nnodes,
    RandomGraphSpecs.Nedges: nedges,
    RandomGraphSpecs.Probability: 0.4,
    RandomGraphSpecs.Seed: seed ,
    RandomGraphSpecs.Repetitions: realrepet,
    RandomGraphSpecs.Distribution: random.gauss,  # random.uniform,
    # RandomGraphSpecs.DistParams: {"a": 1, "b": 30}
    RandomGraphSpecs.DistParams: {"mu": 0.5, "sigma": 0.02}
}

edges, costs0 = gametable.table(TableType.random_graph, graphspecs)  ##TableType.random_tree, graphspecs)

costs0 =[[cost[edge] for edge in edges] for cost in costs0]
costs = costs0[0:repetitions]
coststest = costs0[repetitions:realrepet]

print(edges)
nedges = len(edges)




D_in = nedges
D_out = D_in
specs = [
#     ("relu",nedges*4),
("relu",nnodes),
("relu",nnodes),
("relu",nnodes),
("relu",round(nnodes/10)),

("relu",round(nnodes/10)),
("relu",nnodes),
("relu",nnodes),
("linear", D_out)
]
criterion = "mse"
optimizer = "adam"#
optspecs = { "lr" : 1e-1}#, "momentum": 0.1, "nesterov": True }


autoencoder = Model(D_in = D_in, specs = specs, edges = edges, nnodes = nnodes)
autoencoder.set_loss(type=criterion)
autoencoder.set_optimizer(optimizer, optspecs)

nepochs = 200
MAXEPOCHS = nepochs**2




batchsize = 15

nep = 0
rnep = 0
pregain = fi.max - 1
nepcheck = True
testres = []
while nepcheck:
    rnep +=1
    print("EPOCH", rnep)
    print("___________")
    indices = random.sample(range(repetitions), repetitions)
    xx = [costs[i] for i in indices]
    bmax = round(repetitions/batchsize) - 1
    gain = 0
    for b in range(bmax):
        xxt = torch.as_tensor(xx[b * batchsize : (b + 1) * batchsize])
        y_pred = autoencoder.coremdl(xxt)
        loss = autoencoder.criterion(y_pred, xxt)
        autoencoder.optimizer.zero_grad()
        loss.backward()
        autoencoder.optimizer.step()
        lval = loss.item()
        gain += lval
        #if b%100==0:
        #    print("B", b*batchsize, "loss", lval)
    #print( "pregain",pregain,"gain",gain)
    xxF = torch.as_tensor(coststest)
    yyF = autoencoder.coremdl(xxF)
    lossF = autoencoder.criterion(yyF, xxF)
    ttres = lossF.item()
    print("TTRES",ttres)
    testres.append(ttres)
    nep = nep + 1
    if nep >= nepochs:
        if gain < pregain:
            nep = nep - 5
            #print("here")
        else:
            nepcheck = False
    if rnep >= MAXEPOCHS:
        nepcheck = False
    if pregain > gain:
        pregain = gain

plt.plot(testres)
plt.show()


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
    print(torch.as_tensor([costs[i]]).tolist())
    print(autoencoder.coremdl(torch.as_tensor([costs[i]])).tolist())

xxF = torch.as_tensor(costs)
y_pred = autoencoder.coremdl(xxF)
lossF = autoencoder.criterion(y_pred, xxF)
print("F", lossF.item())

