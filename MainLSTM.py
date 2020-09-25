#%%
import torch
from game import gametable, TableType, RandomJobsSpecs
from BA2 import basicalgo
import networkx as nx
import matplotlib.pyplot as plt
from support import baseline
import random
from environment import EnvSpecs, EnvType
import time
from randomness import randomness, ExplorationSensitivity
import pandas as pd
import numpy as np
from model import Model

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def test(algo, testcosts, noperations, basevals, basetimes):
    stats = algo.test(testcosts, maximumiter=noperations)

    final_objectives = [stat["final_objective"] for stat in stats]
    times = [stat["time"] for stat in stats]

    vy = [-f / g for f, g in zip(final_objectives, basevals)]
    vx = [f / (g + 1e-8) for f, g in zip(times, basetimes)]
    return vx, vy

dtype = torch.float
device = torch.device("cpu")
seed = 1000
random.seed(a=seed)

#Generate random table
data=np.array([])
njobs=10
nmachines=4
# for i in nmach:
#     nmachines=i
repetitions=3

nmachines=int(nmachines)
njobs=int(njobs)
jobsspecs = {
    RandomJobsSpecs.Njobs : njobs,
    RandomJobsSpecs.Nmachines : nmachines,
    RandomJobsSpecs.Probability: 0.7,
    RandomJobsSpecs.Seed : seed,
    RandomJobsSpecs.Repetitions: repetitions*3,
    RandomJobsSpecs.Distribution: random.gauss,
    RandomJobsSpecs.DistParams: {"mu": 100, "sigma": 10}
}

jobs, costs = gametable.table(TableType.random_jobs, jobsspecs) ##TableType.random_tree, graphspecs)
print(jobs, costs)
testcosts = costs[repetitions: repetitions*3]
costs = costs[0:repetitions]
njobs = len(jobs)

# f = pd.read_fwf(r'C:\Users\Marta\Desktop\TESI\istanzagiocattolo.txt', header=None, sep='  ')  #INSERIRE IL PATH
# t = np.array(f)
# jobs=[[job[i] for i in range(0,len(job),2) if not np.isnan(job[i])] for job in t]
# costs=[{job[i]: job[i+1] for i in range(0,len(job),2) if not np.isnan(job[i])} for job in t]
# costs=[costs, costs, costs]
# njobs=len(jobs)
# nmachines=max([len(job) for job in jobs])

basevals = []
basesols = []
for rep in range(repetitions):
    baselin = baseline()
    baseval, basesol, faketime = baselin.js_jobs(jobs, costs[rep])
    #print("    BASELINE", basesol, " BASELINE", -baseval)
    print(" BASELINE", -baseval)
    basevals.append(baseval)
    basesols.append(basesol)
   # basevals.append(1)
   # basesols.append([0,14])
#%%
# define the structure of the network
D_in = 2 #(1, njobs, 2) #batch, seq_lenght, num_features
modelspecs = (nmachines*2,1,1,1) #hidden_dim, layer_dim, output_dim, layer2_dim
#modelspecs = (nmachines*2,1,1,nmachines*2*njobs) #hidden_dim, layer_dim, output_dim, layer2_dim

criterion = "mse"
optimizer = "adam"
optspecs = { "lr" : 1e-4}#, "momentum": 0.1, "nesterov": False }
scheduler = None #"multiplicative"
schedspecs = None #{"factor":0.85}

# launch the algorithm with many repetitions
memorylength = 100000
nepisodes = 10000
memorypath = None
stop_function = None

nop=[(len(job)) for job in jobs]
noperations=sum(nop)

environment_specs = {
    EnvSpecs.type : EnvType.js_LSTM,
    EnvSpecs.statusdimension : D_in,
    EnvSpecs.actiondimension : noperations,
    EnvSpecs.rewardimension : modelspecs[2],
    #EnvSpecs.edges : edges.copy(),
    EnvSpecs.costs : costs.copy(),
    EnvSpecs.prize : 1100,
    EnvSpecs.penalty : -1000,
    #EnvSpecs.finalpoint : nnodes-1,
    #EnvSpecs.startingpoint : 0
    EnvSpecs.jobs : jobs
}

balgo = basicalgo(environment_specs=environment_specs,
                 D_in=D_in,
                 modelspecs = modelspecs,
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
               noptsteps = 1,
               display = (True, 10, True),
               randomness = randomness(r0=1, rule=ExplorationSensitivity.linear_threshold, threshold=0.02, sensitivity=0.999),
               batchsize = 32,
               maximumiter = noperations*2,
               steps = 0,
               backcopy=30)

#print(stats)
final_objectives = [stat["final_objective"] if stat["is_final"] == 1 else 0 for stat in stats ]
plotbasevals = [[-baseval for i in range(len(final_objectives))] for baseval in basevals ]


for plotbase in plotbasevals:
    plt.plot(plotbase)
plt.show()
plt.cla()
path = ''
plt.savefig(path + "Istanza" + str(njobs) + "_" + str(nmachines)+"_LSTM_"+str(nepisodes)+"a"+".jpg")
for plotbase in plotbasevals:
    plt.plot(plotbase)
plt.plot(final_objectives,'o')
path = ''
plt.savefig(path + "Istanza" + str(njobs) + "_" + str(nmachines)+"_LSTM_"+str(nepisodes)+"b"+".jpg")
plt.show()
plt.cla()

gaps = [abs((stat["final_objective"]+basevals[stat["rep"]])/basevals[stat["rep"]])   for stat in stats if stat["is_final"] == 1]
bottom = [0 for _ in  range(len(gaps))]

#Initial and final Loss
Loss_in=stats[15]["Loss_in"]
Loss_fin=stats[nepisodes-1]["Loss_fin"]

# modelpath = "/Users/Marta/Desktop/TESI/model.txt"
# model=Model(D_in,modelspecs,LSTMflag=True)
#
# torch.save({
#     'randomness': randomness,
#      'optspecs' : optspecs,
#     'optimizer' : optimizer,
#     'model_state_dict': model.coremdl.state_dict(),
# }, modelpath)
#
# net=model.coremdl
#
# checkpoint = torch.load(modelpath)
# net.load_state_dict(checkpoint['model_state_dict'])
# optspecs = checkpoint['optspecs']
# randomness = checkpoint['randomness']
# optimizer = checkpoint['optimizer']
#
# checkpoint=torch.load(modelpath)
# net.eval()
#%%
plt.plot(gaps)
plt.plot(bottom)
path = ''
plt.savefig(path + "Istanza" + str(njobs) + "_" + str(nmachines)+"_LSTM_"+str(nepisodes)+"c"+".jpg")
plt.show()
plt.cla()
#noperations = len(operations)
basevals = []
basesols = []
basetimes = []

for rep in range(repetitions*2):
    baselin = baseline()
    baseval, basesol, basetime = baselin.js_jobs(jobs, testcosts[rep])
    basevals.append(baseval)
    basesols.append(basesol)
    basetimes.append(basetime)
    # basevals.append(1)
    # basesols.append([0,14])
    # basetimes.append(1)
#%%
odx, ody = test(balgo, testcosts, noperations*2, basevals, basetimes)
print(odx, ody)
plt.scatter(x=odx, y=ody)
path = ''
plt.savefig(path + "Istanza" + str(njobs) + "_" + str(nmachines)+"LSTM_"+str(nepisodes)+"d"+".jpg")
plt.show()
plt.cla()

stats_test = balgo.test(testcosts, maximumiter=noperations*2)

#MEAN BASETIME
#basetimes1=reject_outliers(np.asarray(basetimes))
meantime_test_CPLEX=(np.mean(basetimes))
#MEAN TIME
times = [stat["time"] for stat in stats_test]
#times1=reject_outliers(np.asarray(times))
meantime_test=np.mean(times)

gaps_test = [(abs(stat["final_objective"] + basevals[stat["rep"]]) / abs(basevals[stat["rep"]])) for stat in stats_test if stat["is_final"] == 1]
mean_error = np.mean(gaps_test)

final_objectives = [stat["final_objective"] for stat in stats_test if stat["is_final"] == 1]
gaps_test1 = [abs(f+g)/abs(g) for f,g in zip(final_objectives, basevals)]
mean_error1 = np.mean(gaps_test1)

#obj_values=[stat["final_objective"] for stat in stats_test if stat["is_final"] == 1]
#df = pd.read_excel("C:/Users/Marta/Desktop/TESI/Stat.xlsx")
row=[njobs, nmachines, memorylength, Loss_in, Loss_fin, meantime_test, meantime_test_CPLEX, mean_error1]
data1=np.array([])
for i in range(repetitions*2):
    row1=[times[i],basetimes[i],final_objectives[i],basevals[i]]
    data1=np.append(data1, row1)

data=np.append(data,row)
# df=pd.DataFrame()
# istances = {'Njobs': njobs, 'Nmachines': nmachines, 'Initial loss': Loss_in, 'Final loss': Loss_fin,
columns = ['Njobs', 'Nmachines', 'Memorylenght','Initial loss', 'Final loss', 'Mean time on test', 'Mean basetime on test',
        'Mean error on obj function test']
columns1 = ['Times', 'Basetimes','Obj values test', 'Obj basevalues test']
data=data.reshape(-1,8)
data1=data1.reshape(-1,4)
df1 = pd.DataFrame(data, columns=columns)
df2 = pd.DataFrame(data1, columns=columns1)
dftot=pd.concat([df1,df2], axis=0, sort=False)
#df.append(dftot)
dftot.to_excel("Stat2.xlsx", index=False, header=True)
#dftot.to_csv("C:/Users/Marta/Desktop/TESI/Stat.csv", index=False, header=True)
