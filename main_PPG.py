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
#torch.set_num_threads(4)
nnodes = 10
nedges = nnodes*(nnodes-1)
repetitions = 1

graphspecs = {
    RandomGraphSpecs.Nnodes : nnodes,
    RandomGraphSpecs.Nedges : nedges,
    RandomGraphSpecs.Probability: None,
    RandomGraphSpecs.Seed: 1,
    RandomGraphSpecs.Repetitions: 3*repetitions,
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
    EnvSpecs.costs : costs.copy(),
    EnvSpecs.prize : 1,
    EnvSpecs.penalty : 0,
    EnvSpecs.finalpoint : nnodes-1,
    EnvSpecs.startingpoint : 0
}

specsActor = {}
specsCritic = {}

specsActor['conv_layers'] = nn.Sequential(nn.Conv2d(1,4,3,stride=1,padding=1),nn.ReLU(),
                                            nn.Conv2d(4,8,3,stride=1,padding=1))
maxfcL = int(specsActor['conv_layers'][-1].out_channels*(nnodes+1)**2)
specsActor['fc_layers'] = nn.Sequential(nn.Linear(maxfcL,30),nn.ReLU(),
                                        #nn.Linear(300,50),nn.ReLU(),
                                        nn.Linear(30,nnodes+1))

specsActor['eps'] = 0.2
specsActor['beta'] = 1
specsActor['beta_c'] = 0.01
specsActor['lr'] = 1e-6
specsActor['maskdim'] = nnodes+1

specsCritic['conv_layers'] = nn.Sequential(nn.Conv2d(1,4,3,stride=1,padding=1),nn.ReLU(),
                                            nn.Conv2d(4,8,3,stride=1,padding=1))
specsCritic['fc_layers'] = nn.Sequential(nn.Linear(maxfcL,30),nn.ReLU(),
                                        nn.Linear(30,1))
specsCritic['lr'] = 5e-4

n_phases = 1500
algo_ppg = ppg(phases=n_phases,policy_iterations=1,
    specsActor=specsActor,
    specsCritic=specsCritic,
    E_policy=1, #Numero di epoche di training in una policy iteration
    E_value=1,
    E_aux=1, #Epoche di training ausiliario
    stacklenght=50000,
    seed=1,
    batchsize=nnodes+1,
    exper=20,
    gamma=1,
    lam=0) #numero di rollout

actor,critic,stats_reward,Loss_actor_stats,Loss_critic_stats,Loss_joint_stats,Loss_aux_stats = algo_ppg.PPG_algo(environment_specs)
import matplotlib.pyplot as plt
def grafico_training(phases,elemento,eltype,num_nodes,num_rollut,opt=None):
    l = [k for k in range(1,phases+1)]
    plt.figure()
    for j in l:
        plt.plot(j,elemento[j-1],'b.',markersize=10)
        if opt is not None:
            plt.plot(j, opt, 'r.', markersize=10)

    plt.xlabel('Phase')
    plt.ylabel(eltype)
    title = 'Numero nodi: '+str(num_nodes)+'   Numero rollout: '+str(num_rollut)
    plt.title(title)
    plt.show()

aaa = baseline_ortools(dis_m=costs,n=nnodes)
opt,time,out = aaa.solve_ortools()
grafico_training(n_phases,stats_reward,'Avg phase rewards',nnodes,10,-sum(opt)/len(opt))
#grafico_training(n_phases,Loss_actor_stats,'Loss Actor')
#grafico_training(n_phases,Loss_critic_stats,'Loss Critic')
#grafico_training(n_phases,Loss_joint_stats,'Loss Joint')
