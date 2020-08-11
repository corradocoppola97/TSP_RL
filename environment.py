import copy
from enum import Enum
# from sys import float_info as fi
import pandas as pd
import numpy as np

f=pd.read_fwf(r'C:\Users\Marta\Desktop\TESI\abz7.txt', header=None, sep='  ')
table=np.array(f)
table=table.reshape(f.shape[0],int(f.shape[1]/2),2)

class EnvType(Enum):
    min_path = 1
    min_path_compressed = 2
    job_shop_scheduling = 3
    train_dispatching = 4
    bnb = 5

class EnvSpecs(Enum):
    # always present
    type = 1
    statusdimension = 2
    actiondimension = 3
    rewardimension = 4
    costs = 5
    prize = 6
    penalty = 7
    # may vary
    # min_path
    edges = 8
    finalpoint = 9
    startingpoint = 10
    # bnb
    As = 11
    bs = 12
    cs = 13
    N = 14
    #job_shop_scheduling
    table = 15

class RewType(Enum):
    linear = 1

class environment():
    def __init__(self, envspecs):
        self._type = envspecs.get(EnvSpecs.type)
        self._statusdimension = envspecs.get(EnvSpecs.statusdimension)
        self._actiondimension = envspecs.get(EnvSpecs.actiondimension)
        self._rewarddimension = envspecs.get(EnvSpecs.rewardimension)
        self._costs = envspecs.get(EnvSpecs.costs)
        self._prize = envspecs.get(EnvSpecs.prize)
        self._penalty = envspecs.get(EnvSpecs.penalty)
        return

    def initial_state(self):
        pass
    def initial_mask(self, st0):
        pass
    def set_instance(self, rep):
        pass
    def instances(self, st, mask):
        pass
    def last_states(self):
        pass
    def output(self, st, at1):
        pass
    def prize(self):
        return self._prize
    def penalty(self):
        return self._penalty
    def linear_costs(self):
        return self._costs

#Mi sono creata un file dalle istanze che mi avevi mandato e l'ho modificato per fare il caso più generale possibile (ti mando un file per farti capire come è costruito)
#ovvero i job hanno un numero di operazioni <= numero macchine, hp: no preemption and jobs do not recirculate)

class job_shop_scheduling(environment):
    def __init__(self, envspecs):
        super(min_path, self).__init__(envspecs)
        self._table = envspecs.get(EnvSpecs.table)  #tensore formato da njobs righe e nmach colonne e elem=(macchina, processing time)
        self._njobs = self._table.shape[0]   #numero dei job
        self._nmach = self._table.shape[1]   #numero delle macchine
        self._operations = [[i] for i in range(self._njobs*self._nmach)] #numeriamo le operazioni (partiamo da 0 per semplicità) + macchina
        for i in range(self._njobs):
            for j in range(self._nmach):
                if (np.isnan(table[i,j,0])):
                  self._operations[(self._nmach)*i+j]=np.nan
                else:
                    (self._operations[(self._nmach)*i+j]).append(int(self._table[i,j,0]))
        #Modo alternativo per operazioni
       # self._ operations = [] #numeriamo le operazioni (partiamo da 0 per semplicità) + macchina
       # count=0
       # for i in range(self._njobs):
       #     for j in range(self._nmach):
       #          if (np.isnan(table[i,j,0])):
       #             self._operations.append(np.nan)
       #          else:
       #             self._operations.append([count]+[int(table[i,j,0])])
       #             count+=1
        self._nop = len(self._operations)   #numero delle operazioni
        self._processingtime = []           #abbino ad ogni operazione, indicata attraverso la posizione della lista, il suo pt
        for i in range(self._njobs):
            for j in range(self._nmach):
                self._processingtime.append(self._table[i,j,1])
        self._prec = []    #lista dei precedenti [op,macchina], se un'op non ce li ha -> []
        for i in range(self._nop):
            if np.any(np.isnan(self._operations[i])):
                self._prec.append(np.nan)
            elif i%self._nmach != 0:   #le operazioni iniziali di ogni job non hanno precedenti
                self._prec.append(self._operations[i-1])
            else:
                self._prec.append([])

    def initial_state(self):   #lista di tante componenti quanto operazioni, ma con istanti di tempo, inizializziamo a -1
        st0=[-1 for op in self._operations]
        for i in range(self._nop):
            if np.any(np.isnan(self._operations[i])):
                st0[i]=np.nan
        return st0

    def initial_mask(self, st0):  #la maschera è una lista di tuple a due elementi: [numero operazione possibile, primo istante in cui si può allocare]
        m0 = []
        for i in range(self._nop):
            if prec[i] == []:
                m0.append(self._operations[i]+[0])
        return m0

    def cost(self, op, st):  #tempo di completamento  (no preemption)
        return st[op] + self._processingtime[op]

    def instances(self, st, mask):
        insts = {}
        states = {}
        at = [-1 for op in range(self._nop)] #definita come one-hot   #? metto nan per coerenza?
        for m in mask:
            at1 = copy.deepcopy(at)
        at1[m[0]] = m[2]
        st1 = copy.deepcopy(st)
        st1[m[0]] = m[2]
        states[m[0]] = st1
        insts[m[0]] = st + at1 + st1  # + self._costlist
        self._last_states = states
        self._insts = insts
        return insts

    def last_states(self):
        return self._last_states

    def output(self, st, at1, last_states=None, insts=None):
        if last_states is not None:
            self._last_states = last_states
        if insts is not None:
            self._insts = insts
        allocated = []
        for i in range(self._nop):
            if st[i] >= -1 + (1e-8):
                allocated.append(self._operations[i])
        #allocated = set(allocated)  #per come definito, non dovrebbe essere necessario
        st1 = self._last_states[at1]
        c = cost(self, at1[0], st)  # at1 in questo caso è una lista due elementi: [op,istante inizio]
        if c < RM:
            rt = -(c + RM)
        else:
            rt = 0
        allocated1=np.array(allocated) #trasformiamo allocated in un array, per sfruttare la funzione np.where
        m = []
        for i in range(self._nop):
            if ((not np.any(np.isnan(self._operations[i]))) and (self._operations[i] not in allocated)): #per tutte le operazioni non nan e non già allocate
                index=np.where(allocated1[:,1]==self._operations[i][1])[-1] #indice in allocated dell'ultima operazione allocata con la stessa macchina
                if index.size>0:  #se c'è un elemento in allocated che usa la stessa macchina di quell'operazione
                    if self._prec[i]==[]:  #e se questa operazione non ha precedenti
                        m.append(self._operations[i] +[cost(allocated[int(index)][0], st)])  #aggiungiamola alla maschera ma il primo istante posibile sarà quello dell'ultima operazione su quella macchina
                    if  self._prec[i] in allocated:  #se l'operazione invece ha precedenti in allocated
                        maxcost=max(cost(allocated1[int(index)][0],st1),cost(self._prec[i][0], st))  #(prendo il tempo di completamento maggiore tra quello del precedente e quello dell'ultima op sulla stessa macchina)
                        m.append(self._operations[i] + [maxcost])
                else:           #se non c'è alcuna operazone già allocata che usa la stessa macchina
                     if self._prec[i] == []:
                       m.append(self._operations[i] + [0])    #disponibile istante 0
                     if self._prec[i] in allocated:
                       m.append(self._operations[i]+[cost(self._prec[i][0], st)])  #disponibile al tempo di completamento del precedente

        final = (-1 not in st)  #quando tutte le operazioni sono state già allocate
        # feasible = len(mask) > 0 or final   #per come definito, non dovrebbero esserci problemi di inammisibilità
        inst = self._insts[at1[0]]
        return st1, rt, final, mask, feasible, inst

class min_path(environment):
    def __init__(self, envspecs):
        super(min_path, self).__init__(envspecs)
        self._edges = envspecs.get(EnvSpecs.edges)
        self._startingpoint = envspecs.get(EnvSpecs.startingpoint)
        self._finalpoint = envspecs.get(EnvSpecs.finalpoint)
        self._nodes = [i for i in range(self._startingpoint, self._finalpoint+1)]
        self._neighbors = [[] for _ in self._nodes]
        for i in self._nodes:
            count = 0
            for (h, k) in self._edges:
                if h == i:
                    self._neighbors[i].append(count)
                count += 1

    def initial_state(self):
        return [0 for edge in self._edges]

    def initial_mask(self, st0):
        return self._neighbors[0]

    def set_instance(self, rep, testcosts=None):
        if testcosts is None:
            self._linear_reward = {key: -self._costs[rep][key] + 0.0 for key in self._costs[rep].keys()}
            self._costlist = list(self._costs[rep].values())
        else:
            self._linear_reward = {key: -testcosts[rep][key] + 0.0 for key in testcosts[rep].keys()}
            self._costlist = list(testcosts[rep].values())

    def instances(self,st, mask):
        insts = {}
        states = {}
        at = [0 for elem in range(len(self._edges))]
        for m in mask:
            #(i, j) = self._edges[m]
            at1 = copy.deepcopy(at)
            at1[m] = 1
            st1 = copy.deepcopy(st)
            st1[m] = 1
            states[m] = st1
            insts[m] = st + at1 + st1 + self._costlist
        self._last_states = states
        self._insts = insts
        return insts

    def last_states(self):
        return self._last_states

    def output(self,st, at1, last_states=None, insts = None):
        if last_states is not None:
            self._last_states = last_states
        if insts is not None:
            self._insts = insts
        visited = [0]
        for i in range(len(st)):
            if st[i] >= 1e-8:
                visited.append(self._edges[i][1]) #self._edges[i] = (h,k) self._edges[i][1] = k
        visited = set(visited)
        st1 = self._last_states[at1]
        rt = -self._costlist[at1]
        nextpoint = self._edges[at1][1]
        mask = []
        for neigh in self._neighbors[nextpoint]:
            (i,j) = self._edges[neigh]
            if j not in visited:###
                mask.append(neigh)
        final = nextpoint == self._finalpoint
        feasible = len(mask) > 0 or final
        inst = self._insts[at1]
        return st1, rt, final, mask, feasible, inst


class testinstance():
    def __init__(self,c,A,b):
        self.A = A
        self.c = c
        self.b = b

class bnb1(environment):
    def __init__(self, envspecs):
        super(bnb1, self).__init__(envspecs)
        self._As = envspecs[EnvSpecs.As]
        self._bs = envspecs[EnvSpecs.As]
        self._cs = envspecs[EnvSpecs.As]
        self._N = envspecs[EnvSpecs.N]


    def initial_state(self):
        return [0 for i in range(self._N*2)]
    def initial_mask(self, st0):
        return [i for i in range(self._N * 2)]
    def set_instance(self, rep, testcosts=None):
        if testcosts is None:
            self._c = self._cs[rep]
            self._A = self._As[rep]
            self._b = self._bs[rep]
        else:
            self._c = testcosts[rep].c
            self._A = testcosts[rep].A
            self._b = testcosts[rep].b


    def instances(self, st, mask):
        insts = {}
        states = {}
        at = self.initial_state().copy()
        for m in mask:
            # (i, j) = self._edges[m]
            at1 = copy.deepcopy(at)
            at1[m] = 1
            st1 = copy.deepcopy(st)
            st1[m] = 1
            states[m] = st1
            insts[m] = st + at1 + st1 + self._c + sum(self._A[i] for i in range(len(self._A))) + self._b
        self._last_states = states
        self._insts = insts
        return insts
    def last_states(self):
        return self._last_states
    def output(self,st, at1, last_states = None, insts = None, bound = None):
        if last_states is not None:
            self._last_states = last_states
        if insts is not None:
            self._insts = insts
        st1 = self._last_states[at1]
        inst = self._insts[at1]

        m1 = [i for i in range(self._N) if st1[i] == 0]
        m2 = [i for i in range(self._N, self._N*2) if st1[i] == 0]
        nodesnotfixed = list(set(m1).intersection(set(m2)))
        mask = nodesnotfixed + [m + self._N for m in nodesnotfixed]
        rt = 0
        if at1 < self._N :
            fixedone = [i for i in  range(self._N) if st1[i] == 1]
            AA = [sum(self._A[j][f] for f in fixedone) for j in range(len(self._b))]
            rt =  -self._c[at1] - self._penalty*sum(AA[j] <= self._b[j] for j in range(len(self._b))) + self._penalty*sum(AA[j]-self._A[j][at1] <= self._b[j] for j in range(len(self._b)))
        final = len(nodesnotfixed) == 1
        feasible = True

        return st1, rt, final, mask, feasible, inst



