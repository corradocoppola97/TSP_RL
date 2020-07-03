# the environment depends on the application
# here actionset feasibility rewards and all the other stuff is organized
from enum import Enum


class EnvType(Enum):
    minpath = 1
    min_path_compressed = 2
    job_shop_scheduling = 3
    train_dispatching = 4

class RewType(Enum):
    linear = 1



class environment():
    def __init__(self, actiondimension, statusdimension, rewardimension, type = EnvType.minpath):
        self._actiondimension = actiondimension
        self._statusdimension = statusdimension
        self._rewardimension = rewardimension
        self._type = type

    def set_minpath(self, finalpoint, edges, prize = 100, penalty = -100, nullalternative = None):
        self._finalpoint = finalpoint
        self._prize = prize
        self._edges = edges
        self._penalty = penalty
        self._multiple_responses = False
        self._nullCheck = nullalternative is not None
        self._nullAlternative = nullalternative
    def set_minpath_compressed(self, finalpoint, edges, prize = 100, penalty = -100, nullalternative = None):
        self.set_minpath( finalpoint, edges, prize , penalty , nullalternative )

    def set_multiple_respose(self):
        self.RS = {edge: 0.0 for edge in self._edges}
        self._multiple_responses = True

    def set_linear_reward(self, c):
        self._c = {key: -c[key]+0.0 for key in c.keys()}
        self._costlist = list(c.values())
        self._reward_type = RewType.linear

    def response(self, st, at): #todo this really depends on the application
        if self._type == EnvType.minpath :
            st1, neighbors1 = self._step_min_path(st, at)
            if self._multiple_responses:
                rs = self.RS.copy()
            if len(neighbors1) == 0 and at[1] != self._finalpoint and not self._multiple_responses:
                return st1, self._penalty, True, neighbors1, False
            elif len(neighbors1) == 0 and at[1] != self._finalpoint and  self._multiple_responses:
                rs[at] = self._penalty
                return st1, (self._penalty, rs), True, neighbors1, False
            rt, final = self._reward(at)

            if not self._multiple_responses:
                return st1, rt, final, neighbors1, True
            else:
                rs[at] = rt + 0.0
                return st1, (rt, rs), final, neighbors1, True
        elif self._type == EnvType.min_path_compressed:
            st1, neighbors1 = self._step_min_path(st, at)
            if len(neighbors1) == 0 and at[1] != self._finalpoint:
                return st1, self._penalty, True, neighbors1, False
            rt, final = self._reward(at)
            return st1, rt, final, neighbors1, True



        else:
            raise Exception("ERROR environment type not defined")



    def _reward(self, at ):
        final = False
        if at[1] == self._finalpoint:
            rt = self._c[at] + self._prize + 0.0
            final = True
        else:
            rt = self._c[at] + 0.0
        return rt, final
    def prize(self):
        return self._prize + 0.0

    def _step_min_path(self, st, at):
        sst = st.copy()
        sst[at] = 1.0
        neighbors = self._neighbors(st, at)
        return sst, neighbors
    def _step_min_path_compressed(self, st, at):
        sst = st.copy()
        sst[at] = self._c[at] + 0.0
        neighbors = self._neighbors_compressed(st, at)
        return sst, neighbors

    def _neighbors_compressed(self, st, atm1):
        connections = [self._edges[i] for i in range(len(self._edges)) if self._edges[i][0] == atm1[1]]
        already_in_the_path = [edge for edge in self._edges if abs(st[edge]) >= 0.01]
        neighbors = list(set(connections) - set(already_in_the_path))
        if self._nullCheck and len(neighbors) >= 1:
            neighbors = neighbors + [self._nullAlternative]

    def _neighbors(self, st, atm1):
        connections = [self._edges[i] for i in range(len(self._edges)) if self._edges[i][0] == atm1[1]]
        already_in_the_path = [edge for edge in self._edges if 1.0 <= abs(st[edge]) <= 1.0]
        neighbors = list(set(connections) - set(already_in_the_path))
        if self._nullCheck and len(neighbors) >= 1:
            neighbors = neighbors + [self._nullAlternative]
        return neighbors
    def neighbors0(self, atm1):
        neighbors = [self._edges[i] for i in range(len(self._edges)) if self._edges[i][0] == atm1[1]]
        return neighbors

    def action_minpath(self, qvalues, neighbors):
        indmax = qvalues.index(max(qvalues))
        return neighbors[indmax]

    def action_minpath_compressed(self, qvalues, neighbors):
        indmax = qvalues.index(max(qvalues))
        return neighbors[indmax]

    def action(self, qvalues, moves):
        if self._type == EnvType.minpath:
            return self.action_minpath(qvalues, moves)
        elif self._type == EnvType.min_path_compressed:
            return self.action_minpath_compressed(qvalues, moves)
        else:
            raise Exception("ERROR environment not supported.")

    def Odin_action(self, qvalues, anotherqvalues, neighbors):
        if self._type == EnvType.min_path_compressed or self._type == EnvType.minpath:
            qmax = max(qvalues)
            anqmax = max(anotherqvalues)
            indmax = qvalues.index(qmax) if qmax > anqmax else anotherqvalues.index(anqmax)
            return neighbors[indmax]
        else:
            raise Exception("ERROR environment not supported.")


    def shrink_feasible_moves(self, st, at):
        return 0

    def check_feasibility(self, st, at):
        return True

    def encode(self, action):
        code = {}
        if self._type == EnvType.minpath:
            code = {edge: 0.0 for edge in self._edges }
            code[action] = 1.0
        return code
    def decode(self, neighbors):
        code = {}
        for i in range(len(self._edges)):
            for edge in neighbors:
                if self._edges[i] == edge:
                    code[edge] = i
        return code
    def costlist(self):
        return self._costlist
    def penalty(self):
        return self._penalty





