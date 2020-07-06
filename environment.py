import copy
from enum import Enum


class EnvType(Enum):
    min_path = 1
    min_path_compressed = 2
    job_shop_scheduling = 3
    train_dispatching = 4

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
    edges = 8
    finalpoint = 9
    startingpoint = 10

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
    def linear_costs(self):
        return self._costs
    def initial_state(self):
        pass
    def initial_mask(self, st0):
        pass
    def set_linear_reward(self, c):
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

    def set_linear_reward(self, c):
        self._linear_reward = {key: -c[key]+0.0 for key in c.keys()}
        self._costlist = list(c.values())

    def instances(self,st, mask):
        insts = {}
        states = {}
        for m in mask:
            (i, j) = self._edges[m]
            st1 = copy.deepcopy(st)
            st1[m] = 1
            states[m] = st1
            insts[m] = st + [i, j] + st1 + self._costlist
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
                visited.append(self._edges[i][1])
        visited = set(visited)
        st1 = self._last_states[at1]
        rt = self._costlist[at1]
        nextpoint = self._edges[at1][1]
        mask = []
        for neigh in self._neighbors[nextpoint]:
            (i,j) = self._edges[neigh]
            if j not in visited:
                mask.append(neigh)
        feasible = len(mask) > 0
        final = nextpoint == self._finalpoint or not feasible
        inst = self._insts[at1]
        return st1, rt, final, mask, feasible, inst



