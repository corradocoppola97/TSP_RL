from model import PPG_Model
from modules2 import Actor,Critic
from data_manager import data_manager
from Environment_GH import environment, EnvSpecs, EnvType, tsp
import copy
import random
import torch
import torch.nn as nn
from PPG import *
from Game_GH import TableType, RandomGraphSpecs, gametable
from TSP_ortools import risolvi_problema, print_solution
from support import baseline_ortools
torch.set_num_threads(4)
nnodes = 15
nedges = nnodes*(nnodes-1)
repetitions = 1

graphspecs = {
    RandomGraphSpecs.Nnodes : nnodes,
    RandomGraphSpecs.Nedges : nedges,
    RandomGraphSpecs.Probability: None,
    RandomGraphSpecs.Seed: 1,
    RandomGraphSpecs.Repetitions: repetitions,
    RandomGraphSpecs.Distribution: None,
    RandomGraphSpecs.DistParams: (0,100,2)
}

edges, costs1, ds = gametable._random_graph_distances(graphspecs)
testcosts = costs1[repetitions: repetitions*3]
costs = costs1[0:repetitions]
D_in = int((nnodes+1)**2)
D_out = 1

environment_specs = {
    EnvSpecs.type : EnvType.tsp,
    EnvSpecs.statusdimension : D_in,
    EnvSpecs.actiondimension : nedges,
    EnvSpecs.rewardimension : D_out,
    EnvSpecs.edges : edges.copy(),
    EnvSpecs.costs : costs1.copy(),
    EnvSpecs.prize : 1,
    EnvSpecs.penalty : 0,
    EnvSpecs.finalpoint : nnodes-1,
    EnvSpecs.startingpoint : 0
}

specsActor = {}
specsCritic = {}

specsActor['conv_layers'] = nn.Sequential(nn.Conv2d(1,8,3,stride=1,padding=1),nn.ReLU(),
                                            nn.Conv2d(8,32,3,stride=1,padding=1),nn.ReLU(),
                                            nn.Conv2d(32,64,3,stride=1,padding=1))
maxfcL = int(specsActor['conv_layers'][-1].out_channels*(nnodes+1)**2)
specsActor['fc_layers'] = nn.Sequential(nn.Linear(maxfcL,300),nn.ReLU(),
                                        nn.Linear(300,50),nn.ReLU(),
                                        nn.Linear(50,nnodes+1))

specsActor['eps'] = 0.2
specsActor['beta'] = 0.01
specsActor['beta_c'] = 1
specsActor['lr'] = 5e-4
specsActor['maskdim'] = nnodes+1

specsCritic['conv_layers'] = nn.Sequential(nn.Conv2d(1,8,3,stride=1,padding=1),nn.ReLU(),
                                            nn.Conv2d(8,32,3,stride=1,padding=1),nn.ReLU(),
                                            nn.Conv2d(32,64,3,stride=1,padding=1))
specsCritic['fc_layers'] = nn.Sequential(nn.Linear(maxfcL,300),nn.ReLU(),
                                        nn.Linear(300,50),nn.ReLU(),
                                        nn.Linear(50,1))
specsCritic['lr'] = 5e-4

gatto = ppg(phases=1,policy_iterations=10,
    specsActor=specsActor,
    specsCritic=specsCritic,
    E_policy=1000,
    E_value=20,
    E_aux=5,
    stacklenght=50000,
    seed=1,
    batchsize=15)

ppg_al = gatto.PPG_algo(environment_specs)

