from Environment_GH import EnvType, EnvSpecs, RewType, environment, min_path, tsp
from Game_GH import TableType, RandomGraphSpecs, gametable
from modules2 import TSP_Model, Encoder, CNN_st, CNN_st1, MLP
import numpy as np
import torch
import random
nnodes = 5
nedges = 20
repetitions = 1
graphspecs = {
    RandomGraphSpecs.Nnodes : nnodes,
    RandomGraphSpecs.Nedges : nedges,
    RandomGraphSpecs.Probability: None,
    RandomGraphSpecs.Seed: 1,
    RandomGraphSpecs.Repetitions: repetitions,
    RandomGraphSpecs.Distribution: None,
    RandomGraphSpecs.DistParams: None
}
edges, costs,ds = gametable._random_graph_distances(graphspecs)
#edges, costs = gametable.table( TableType.random_graph, graphspecs)
D_in = nnodes*nnodes
D_out = 1
environment_specs = {
    EnvSpecs.type : EnvType.min_path,
    EnvSpecs.statusdimension : D_in,
    EnvSpecs.actiondimension : nnodes,
    EnvSpecs.rewardimension : D_out,
    EnvSpecs.edges : edges.copy(),
    EnvSpecs.costs : costs.copy(),
    EnvSpecs.prize : 1000,
    EnvSpecs.penalty : 0,
    EnvSpecs.finalpoint : nnodes-1,
    EnvSpecs.startingpoint : 0
}


#a = min_path(environment_specs)
a = tsp(environment_specs)
s0 = a.initial_state()
m0 = a.initial_mask(s0)
a.set_instance(0)
insts01 = a.instances(s0,m0)
s1,r1,final1,m1,feasible1,inst1 = a.output(s0,3)
print(a.current_depot)
insts12 = a.instances(s1,m1)
s2,r2,final2,m2,feasible2,inst2 = a.output(s1,m1[2])
print(a.current_depot)
insts23 = a.instances(s2,m2)
s3,r3,final3,m3,feasible3,inst3 = a.output(s2,m2[1])
print(a.current_depot)
insts34 = a.instances(s3,m3)
s4,r4,final4,m4,feasible4,inst4 = a.output(s3,m3[0])
print(a.current_depot)
insts45 = a.instances(s4,m4)
s5,r5,final5,m5,feasible5,inst5 = a.output(s4,m4[0])
print(a.current_depot)

