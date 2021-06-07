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
from PPO import ppo
#torch.set_num_threads(4)
nnodes = 10
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
    EnvSpecs.costs : costs.copy(),
    EnvSpecs.prize : 1,
    EnvSpecs.penalty : 0,
    EnvSpecs.finalpoint : nnodes-1,
    EnvSpecs.startingpoint : 0
}

specsActor = {}
specsCritic = {}

specsActor['conv_layers'] = nn.Sequential(nn.Conv2d(1,4,3,stride=1,padding=1),nn.ReLU(),
                                            nn.Conv2d(4,8,3,stride=1,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,stride=1,padding=1))
maxfcL = int(specsActor['conv_layers'][-1].out_channels*(nnodes+1)**2)
specsActor['fc_layers'] = nn.Sequential(nn.Linear(maxfcL,30),nn.ReLU(),
                                        #nn.Linear(300,50),nn.ReLU(),
                                        nn.Linear(30,nnodes+1))

specsActor['eps'] = 0.2
specsActor['beta'] = 1
specsActor['beta_c'] = 0.01
specsActor['lr'] = 1e-6
specsActor['maskdim'] = nnodes+1

specsCritic['conv_layers'] = nn.Sequential(nn.Conv2d(1,8,3,stride=1,padding=1),nn.ReLU(),
                                            nn.Conv2d(8,8,3,stride=1,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,stride=1,padding=1))
specsCritic['fc_layers'] = nn.Sequential(nn.Linear(maxfcL,30),nn.ReLU(),
                                        nn.Linear(30,1))
specsCritic['lr'] = 1e-3
specsActor['emb_size'] = 16
specsCritic['emb_size'] = 16

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
    exper=10,
    gamma=1,
    lam=0) #numero di rollout

nit = 1500
exper = 10
epochs = 3
algo_ppo = ppo(nit=nit,
            epochs=epochs,
            exper=exper,
            batchsize=nnodes+1,
            beta=1,
            specsActor=specsActor,
            specsCritic=specsCritic)

actor,critic,stats_reward,Loss_actor_stats,Loss_critic_stats,Loss_joint_stats,Loss_aux_stats = algo_ppg.PPG_algo(environment_specs)
import matplotlib.pyplot as plt
#stats_loss_actor_ppo,stats_loss_critic_ppo,stats_reward_ppo = algo_ppo.ppo_algo(envspecs=environment_specs,threshold=0.25)


aaa = baseline_ortools(dis_m=costs,n=nnodes)
opt,time,out = aaa.solve_ortools()




def training_ppg(stats,opt,phases,it,num_rep,exper):
    stats_avg, stats_best = [], []
    for ph in range(phases):
        stats_phase_ph = stats[ph]
        for iteration in range(it):
            stats_iteration = stats_phase_ph[iteration]
            best_rew,avg_rew = [],[]
            for rep in range(num_rep):
                best = max(stats_iteration[rep])
                avg = sum(stats_iteration[rep])/len(stats_iteration[rep])
                best_rew.append(-opt[rep]/best)
                avg_rew.append(-opt[rep]/avg)
        stats_avg.append(avg_rew)
        stats_best.append(best_rew)
    d,f = {}, {}
    for ph in range(phases):
        d[ph] = stats_avg[ph]
        f[ph] = stats_best[ph]

    return d,f

avg, best = training_ppg(stats_reward,opt,n_phases,1,repetitions,algo_ppg.experience_dataset_lenght)

def training_ppo(stats,opt,nit,num_rep,exper,flag):
    d = {}
    if flag == 'avg':
        for i in range(nit):
            stats_i = stats[i]
            avg_rewards_norm = []
            for k in range(num_rep):
                avg_r = sum(stats_i[k])/len(stats_i[k])
                norm_r = -opt[k]/avg_r
                avg_rewards_norm.append(norm_r)
            d[i] = avg_rewards_norm

    if flag == 'best':
        for i in range(nit):
            stats_i = stats[i]
            best_rewards_norm = []
            for k in range(num_rep):
                best_r = max(stats_i[k])
                norm_r = -opt[k]/best_r
                best_rewards_norm.append(norm_r)
            d[i] = best_rewards_norm
    return d

#d = training_ppo(stats_reward_ppo,opt,nit,repetitions,exper,'avg')

def grafico_normalized_reward(dict,flag,LC=None):
    h = len(dict)
    plt.figure()
    if flag == 'avg':
        for i in range(h):
            plt.plot(i+1,sum(dict[i])/len(dict[i]),'b.',markersize=7)
    elif flag == 'best':
        for i in range(h):
            plt.plot(i+1,max(dict[i]),'b.',markersize=7)

    elif flag == 'all':
        for i in range(h):
            for ii in range(len(dict[i])):
                plt.plot(i+1,dict[i][ii],LC[ii],markersize=7)

    plt.xlabel('Phases')
    plt.ylabel('Reward Statistics Type '+flag)
    plt.show()


#grafico_normalized_reward(avg,'avg')








'''
def training_rep(phases,rewards,eltype,opt,rep):
    ott = opt[rep]
    l = [_ for _ in range(phases)]
    plt.figure()
    for j in l:
        plt.plot(j,rewards[j][rep],'b.',markersize=10)
        plt.plot(j,-opt[rep],'r.',markersize=10)

    plt.xlabel('Phase')
    plt.ylabel(eltype)
    plt.show()

def confronto_generale(phases,rewards,opt):
    n_rep = len(opt)
    l = [_ for _ in range(phases)]
    d = {}
    for j in l:
        l_graf = []
        for jj in range(n_rep):
            curr_opt = -opt[jj]
            curr_res = rewards[j][jj]
            norm_res = curr_opt/curr_res
            l_graf.append(norm_res)
        d[j] = l_graf

    return d

def plot_confronto_generale(d,eltype):
    plt.figure()
    for key in d:
        l = d[key]
        plt.plot(key,sum(l)/len(l),'b.',markersize=7)
    plt.xlabel('Phase')
    plt.ylabel(eltype)
    plt.show()
    

def grafico_training(phases,elemento,eltype,num_nodes,num_rollut,opt=None):
    l = [k for k in range(1,phases+1)]
    plt.figure()
    for j in l:
        plt.plot(j,elemento[j-1],'bx',markersize=10)
        if opt is not None:
            plt.plot(j, opt, 'r.', markersize=10)

    plt.xlabel('Phase')
    plt.ylabel(eltype)
    title = 'Numero nodi: '+str(num_nodes)+'   Numero rollout: '+str(num_rollut)
    plt.title(title)
    plt.show()
    return None
    

'''
