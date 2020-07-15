import copy
from enum import Enum
#from sys import float_info as fi


class EnvType(Enum):
    min_path = 1
    min_path_compressed = 2
    job_shop_scheduling = 3
    train_dispatching = 4
    bnb = 5

class EnvSpecs(Enum):
    #always present
    type = 1
    statusdimension = 2
    actiondimension = 3
    rewardimension = 4
    costs = 5
    prize = 6
    penalty = 7
    #may vary
    #min_path
    edges = 8
    finalpoint = 9
    startingpoint = 10
    #bnb
    As = 11
    bs = 12
    cs = 13
    N = 14

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
    def output(self,st, at1):
        pass

    def prize(self):
        return self._prize
    def penalty(self):
        return self._penalty
    def linear_costs(self):
        return self._costs



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



