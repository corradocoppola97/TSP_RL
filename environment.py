import copy
from enum import Enum
# from sys import float_info as fi
import numpy as np

class EnvType(Enum):
    min_path = 1
    min_path_compressed = 2
    job_shop_scheduling = 3
    js_LSTM = 4
    train_dispatching = 5
    bnb1 = 6

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
    operations = 15
    #js_LSTM
    jobs = 16

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

class js_LSTM(environment):
    def __init__(self, envspecs):
        super(js_LSTM, self).__init__(envspecs)
        self._jobs = envspecs.get(EnvSpecs.jobs)

        self._operations = [[(i,j,self._jobs[i][j]) for j in range(len(self._jobs[i]))] for i in range(len(self._jobs))]
        self._prec = [[() if j == 0 else self._operations[i][j - 1] for j in range(len(self._operations[i]))] for i in range(len(self._operations))]

    def initial_state(self):  # lista di tante componenti quanto operazioni, ma con istanti di tempo, inizializziamo a -1
        st0 = [[-1 for _ in range(len(self._operations[i]))] for i in range(len(self._operations))]
        return st0

    def initial_mask(self, st0):  # la maschera è una lista di tuple a quattro elementi: [numero, job, macchina, primo istante in cui si può allocare]
        m0 = [self._operations[i][0] + (0,) for i in range(len(self._operations))]
        return m0

    def set_instance(self, rep, testcosts=None):
        if testcosts is None:
            self._linear_reward = [{key: -self._costs[rep][i][key] + .0 for key in self._costs[rep][i].keys()} for i in range(len(self._costs[0]))]
            self._proc_times = [list(self._costs[rep][i].values()) for i in range(len(self._costs[rep]))]
        else:
            self._linear_reward = [{key: -testcosts[rep][i][key] + 0.0 for key in testcosts[rep][i].keys()} for i in range(len(testcosts[0]))]
            self._proc_times = [list(testcosts[rep][i].values()) for i in range(len(testcosts[rep]))]

    def completion_time(self, job, op, st):  # tempo di completamento  (no preemption)
        return st[job][op] + self._proc_times[job][op]

    def instances(self, st, mask):
        insts = {}
        states = {}
        at = [[-1 for _ in range(len(self._operations[i]))] for i in range(len(self._operations))]
        jobs0 = [[(m, t) for m, t in list(zip(self._jobs[i], self._proc_times[i]))] for i in range(len(self._jobs))]
        jobs = [[jobs0[i][j] for j in range(len(st[i])) if st[i][j] == -1] for i in range(len(st))]
        jobs = [job + [(-1, -1)] for job in jobs]
        for m in mask:
            jobs1 = copy.deepcopy(jobs)
            at1 = copy.deepcopy(at)
            at1[m[0]][m[1]] = m[-1]
            st1 = copy.deepcopy(st)
            st1[m[0]][m[1]] = m[-1]
            states[m] = st1
            op=(m[2],jobs0[m[0]][m[1]][1])
            jobs1[m[0]].remove(op)
            insts[m] = jobs1
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
        allocated = [self._operations[i][j] for i in range(len(st)) for j in range(len(st[i])) if st[i][j] >= -1 + (1e-6)]
        st1 = self._last_states[at1]
        C = self.completion_time(at1[0], at1[1], st1)
        rt = 0
        if len(allocated) > 0:
            pippo = [self.completion_time(all[0], all[1], st) for all in allocated]
            if C > max(pippo):
                rt = -(C - max(pippo))
        else:
            rt = -C
        allocated.append(at1[:-1])
        # print("allocated",allocated)
        allocated1 = np.array(allocated)  # trasformiamo allocated in un array, per sfruttare la funzione np.where
        mask = []
        for i in range(len(self._operations)):
          for j in range(len(self._operations[i])):
            if (self._operations[i][j] not in allocated):  # per tutte le operazioni  non già allocate
                index = np.where(allocated1[:, 2] == self._operations[i][j][2])[0]  # indici in allocated delle operazioni allocate con la stessa macchina
                if index.size > 0:  # se c'è un elemento in allocated che usa la stessa macchina di quell'operazione
                    index = index[-1]  # indice in allocated dell'ultima operazione allocata con la stessa macchina
                    if self._prec[i][j] == ():  # e se questa operazione non ha precedenti
                        mask.append(self._operations[i][j] + (self.completion_time(allocated[int(index)][0],allocated[int(index)][1],st1),))  # aggiungiamola alla maschera ma il primo istante possibile sarà quello dell'ultima operazione su quella macchina
                    if self._prec[i][j] in allocated:  # se l'operazione invece ha precedenti in allocated
                        maxtime = max(self.completion_time(allocated[int(index)][0],allocated[int(index)][1], st1),self.completion_time(self._prec[i][j][0],self._prec[i][j][1],st1))  # (prendo il tempo di completamento maggiore tra quello del precedente e quello dell'ultima op sulla stessa macchina)
                        mask.append(self._operations[i][j] + (maxtime,))

                else:  # se non c'è alcuna operazione già allocata che usa la stessa macchina
                    if self._prec[i][j] == ():
                        mask.append(self._operations[i][j] + (0,))  # disponibile istante 0
                    if self._prec[i][j] in allocated:
                        mask.append(self._operations[i][j] + (self.completion_time(self._prec[i][j][0], self._prec[i][j][1],st1),))  # disponibile al tempo di completamento del precedente
        # print("mask",mask)
        final = len(mask) == 0  # quando tutte le operazioni sono state già allocate
        feasible = True
        inst = self._insts[at1]
        # print("at1", at1, "st1", st1)
        return st1, rt, final, mask, feasible, inst

class job_shop_scheduling(environment):
    def __init__(self, envspecs):
        super(job_shop_scheduling, self).__init__(envspecs)
        self._operations = envspecs.get(EnvSpecs.operations)  #lista di tuple, operation=(job, macchina)

        self._operations = [(i,)+self._operations[i] for i in range(len(self._operations))]  #lista di tuple, operation=(numero, job, macchina)
        self._prec = [() if i == 0 or self._operations[i][1] != self._operations[i - 1][1] else self._operations[i-1] for i in range(len(self._operations))]
        #lista dei precedenti [numero, job, macchina], se un'operazione non ha precedenti -> ()

    def initial_state(self):  #lista di tante componenti quanto operazioni, ma con istanti di tempo, inizializziamo a -1
        st0 = [-1 for _ in range(len(self._operations))]
        return st0

    def initial_mask(self,st0):  #la maschera è una lista di tuple a quattro elementi: [numero, job, macchina, primo istante in cui si può allocare]
        m0 = [self._operations[i] + (0,) for i in range(len(self._prec)) if self._prec[i] == ()]
        return m0

    def set_instance(self, rep, testcosts=None):
        if testcosts is None:
            self._linear_reward = {key: -self._costs[rep][key] + 0.0 for key in self._costs[rep].keys()}
            self._proc_times = list(self._costs[rep].values())
        else:
            self._linear_reward = {key: -testcosts[rep][key] + 0.0 for key in testcosts[rep].keys()}
            self._proc_times = list(testcosts[rep].values())

    def completion_time(self, op, st):  # tempo di completamento  (no preemption)
        return st[op] + self._proc_times[op]

    def instances(self, st, mask):
        insts = {}
        states = {}
        at = [-1 for _ in range(len(self._operations))]
        for m in mask:
            at1 = copy.deepcopy(at)
            at1[m[0]] = m[-1]
            st1 = copy.deepcopy(st)
            st1[m[0]] = m[-1]
            states[m] = st1
            insts[m] = st + at1 + st1 + self._proc_times
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
        allocated = [self._operations[i] for i in range(len(st)) if st[i] >= -1 + (1e-6)]
        #allocated = set(allocated)  #per come definito, non dovrebbe essere necessario
        st1 = self._last_states[at1] # at1 in questo caso è una tupla di qattri elementi: (nop,job, macchina, istante inizio), si potrebbe anche ridurre a (nop, istante inzio) con qualche modifica nel codice
        C = self.completion_time(at1[0], st1)
        rt=0
        if len(allocated)>0:
            pippo = [self.completion_time(all[0], st) for all in allocated]
            if C > max(pippo):
                rt = -(C - max(pippo))
        else:
            rt=-C
        allocated.append(at1[:-1])
        #print("allocated",allocated)
        allocated1 = np.array(allocated)  # trasformiamo allocated in un array, per sfruttare la funzione np.where
        mask = []
        for i in range(len(self._operations)):
             # if len(allocated) > 0:
                if (self._operations[i] not in allocated):  # per tutte le operazioni  non già allocate
                        index = np.where(allocated1[:, 2] == self._operations[i][2])[0]  # indici in allocated delle operazioni allocate con la stessa macchina
                        if index.size > 0:  # se c'è un elemento in allocated che usa la stessa macchina di quell'operazione
                           index = index[-1] # indice in allocated dell'ultima operazione allocata con la stessa macchina
                           if self._prec[i] == ():  # e se questa operazione non ha precedenti
                                mask.append(self._operations[i] + (self.completion_time(allocated[int(index)][0],st1),))  # aggiungiamola alla maschera ma il primo istante possibile sarà quello dell'ultima operazione su quella macchina
                           if self._prec[i] in allocated: #se l'operazione invece ha precedenti in allocated
                                maxtime = max(self.completion_time(allocated[int(index)][0], st1),self.completion_time(self._prec[i][0], st1))  # (prendo il tempo di completamento maggiore tra quello del precedente e quello dell'ultima op sulla stessa macchina)
                                mask.append(self._operations[i] + (maxtime,))

                        else:  # se non c'è alcuna operazione già allocata che usa la stessa macchina
                            if self._prec[i] == ():
                                mask.append(self._operations[i] + (0,)) # disponibile istante 0
                            if self._prec[i] in allocated:
                                mask.append(self._operations[i] + (self.completion_time(self._prec[i][0], st1),))  # disponibile al tempo di completamento del precedente
        #print("mask",mask)
        final = (-1 not in st1)  # quando tutte le operazioni sono state già allocate
        feasible = True
        inst = self._insts[at1]
        #print("at1", at1, "st1", st1)
        return st1, rt, final, mask, feasible, inst

#se invece carichiamo istanza da file sottoforma di table
#inserire nel main:
# f = pd.read_fwf(r'C:\Users\Marta\Desktop\TESI\istanza.txt', header=None, sep='  ')
# t = np.array(f)
# table = [[] for i in range(t.shape[0])]
# for i in range(t.shape[0]):
#     for j in range(0, t.shape[1], 2):
#         if not np.isnan(t[i][j]):
#             table[i].append([int(t[i][j])] + [t[i][j + 1]])
#
#inserire nella classe:
        # self._table = envspecs.get(EnvSpecs.table)  # tensore formato da njobs righe e nmach colonne e elem=(macchina, processing time)
        #
        # self._operations = []
        # count = 0
        # for i in range(len(self._table)):
        #     job = i
        #     for j in range(len(self._table[i])):
        #         mach = self._table[i][j][0]
        #         op = (job, mach)
        #         self._operations.append(op)
        #         count += 1
        #
        # self._proc_time = []
        # for i in range(len(self._table)):
        #     for j in range(len(self._table[i])):
        #         self._proc_time.append(self._table[i][j][-1])

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



