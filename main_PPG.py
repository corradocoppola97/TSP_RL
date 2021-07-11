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
from support import baseline_ortools,training_ppo,training_ppg,grafico_normalized_reward,grafico_test
from PPO import ppo
path = '\\Users\corra\OneDrive\Desktop\Tesi\ModelliSalvati'
#torch.set_num_threads(4)
nnodes = 20
nedges = nnodes*(nnodes-1)
repetitions = 30
device = torch.device('cpu')
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
model_max_dim = nnodes+1
specsActor['conv_layers'] = nn.Sequential(nn.Conv2d(1,4,3,stride=1,padding=1),nn.ReLU(), #nn.Conv2d(4,8,3,stride=1,padding=1),nn.ReLU(),
                                            nn.Conv2d(4,8,3,stride=1,padding=1))
maxfcL = int(specsActor['conv_layers'][-1].out_channels*(model_max_dim)**2)
specsActor['fc_layers'] = nn.Sequential(nn.Linear(maxfcL,30),nn.ReLU(),
                                        #nn.Linear(300,50),nn.ReLU(),
                                        nn.Linear(30,nnodes+1))

specsActor['eps'] = 0.2
specsActor['beta'] = 1
specsActor['beta_c'] = 0.01
specsActor['lr'] = 1e-7
specsActor['maskdim'] = nnodes+1

specsCritic['conv_layers'] = nn.Sequential(nn.Conv2d(1,4,3,stride=1,padding=1),nn.ReLU(), #nn.Conv2d(4,8,3,stride=1,padding=1),nn.ReLU(),
                                            nn.Conv2d(4,8,3,stride=1,padding=1))
specsCritic['fc_layers'] = nn.Sequential(nn.Linear(maxfcL,30),nn.ReLU(),
                                        nn.Linear(30,1))
specsCritic['lr'] = 5e-4
specsActor['emb_size'] = 16
specsCritic['emb_size'] = 16
specsCritic['in_channels'] = (nnodes+1)*2
specsActor['in_channels'] = (nnodes+1)*2

n_phases = 300
algo_ppg = ppg(phases=n_phases,policy_iterations=9,
    specsActor=specsActor,
    specsCritic=specsCritic,
    E_policy=1, #Numero di epoche di training in una policy iteration
    E_value=1,
    E_aux=3, #Epoche di training ausiliario
    stacklenght=5000000000,
    seed=1,
    batchsize=nnodes+1,
    exper=10,
    gamma=1,
    lam=0,
    device=device,GCN_flag=False) #numero di rollout

nit = 7500
exper = 10
epochs = 3
algo_ppo = ppo(nit=nit,
            epochs=epochs,
            exper=exper,
            batchsize=nnodes+1,
            beta=1,
            specsActor=specsActor,
            specsCritic=specsCritic,
            device=device,GCNflag=False)

f1 = '0907_30g20n_PPO.txt'
file_name = ('\policy_'+f1,'\pvalue_'+f1)
#actor,critic,stats_reward,Loss_actor_stats,Loss_critic_stats,Loss_joint_stats,Loss_aux_stats,rgs = algo_ppg.PPG_algo(environment_specs,file_name=file_name,threshold=0.25)
import matplotlib.pyplot as plt
#algo_ppo.load_models(path,file_name)
stats_loss_actor_ppo,stats_loss_critic_ppo,stats_reward_ppo,actor_ppo,critic_ppo = algo_ppo.ppo_algo(envspecs=environment_specs,threshold=0.25, file_name=file_name)

aaa = baseline_ortools(dis_m=costs,n=nnodes)
opt,time,out = aaa.solve_ortools()

#avg, best = training_ppg(stats_reward,opt,n_phases,1,repetitions,algo_ppg.experience_dataset_lenght)

d = training_ppo(stats_reward_ppo,opt,nit,repetitions,exper,'avg')
dd = training_ppo(stats_reward_ppo,opt,nit,repetitions,exper,'best')


#grafico_normalized_reward(avg,'avg')

graphspecs_test = {
    RandomGraphSpecs.Nnodes : nnodes,
    RandomGraphSpecs.Nedges : nedges,
    RandomGraphSpecs.Probability: None,
    RandomGraphSpecs.Seed: 1,
    RandomGraphSpecs.Repetitions: 100,
    RandomGraphSpecs.Distribution: None,
    RandomGraphSpecs.DistParams: (0,100,2)
}

edges_test, costs1_test, ds_test = gametable._random_graph_distances(graphspecs_test)
D_in = int((nnodes+1)**2)
D_out = 1

environment_test = {
    EnvSpecs.type : EnvType.tsp,
    EnvSpecs.statusdimension : D_in,
    EnvSpecs.actiondimension : nedges,
    EnvSpecs.rewardimension : D_out,
    EnvSpecs.edges : edges_test.copy(),
    EnvSpecs.costs : costs1_test.copy(),
    EnvSpecs.prize : 1,
    EnvSpecs.penalty : 0,
    EnvSpecs.finalpoint : nnodes-1,
    EnvSpecs.startingpoint : 0
}


#test_stats_ppg = algo_ppg.Test_Policy(actor,critic,0,environment_test)
#actor_ppo,critic_ppo = algo_ppo.load_models(path,file_name)
#actor_ppo, critic_ppo = algo_ppo.load_models(path,file_name)
#test_stats_ppo,times_ppo = algo_ppo.Test_Policy(actor_ppo,critic_ppo,0,environment_test)
test_aaa = baseline_ortools(dis_m=costs1_test,n=nnodes)
opt_test, time_test, out_test = test_aaa.solve_ortools()

#gtt = grafico_test(opt_test,test_stats_ppo)
#print('media',sum(gtt)/len(gtt))
#print('tempi:',max(time_test)/min(times_ppo))


