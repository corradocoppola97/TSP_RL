from model import Model
from data_manager import data_manager
from environment import environment, EnvType, RewType
from enum import Enum
import torch
torch.set_num_threads(4)
import random
import math
from Thor import EnvSpecs
from sys import float_info as fi

class thor2():
    def __init__(self, environment_specs,
                 D_in,
                 modelspecs,
                 criterion = "mse",
                 optimizer = "sgd",
                 optspecs = {"lr":1e-4},
                 memorylength = None,
                 memorypath = None,
                 seed = None,
                 stop_function = None): #todo implement stopping functions
        self._model = Model(D_in, modelspecs, seed)
        self._model.set_loss(criterion)
        self._model.set_optimizer(name=optimizer, options=optspecs)
        self._data  = data_manager(stacklength=memorylength, seed=seed)
        if memorypath != None:
            self._data.memoryrecoverCSV(memorypath)
        self._stop_function = stop_function
        if stop_function == None:
            self._stop_function = thor2._no_stop
        self._envspecs = environment_specs


    def _no_stop(elem):
        return False

    def _buildenvironment(self, envspecs):
        if envspecs[EnvSpecs.type] == EnvType.minpath:
            self._type = EnvType.minpath
            actiondimension =  envspecs[EnvSpecs.actiondimension]
            statusdimension =  envspecs[EnvSpecs.statusdimension]
            rewardimension  =  envspecs[EnvSpecs.rewardimension]
            self.env = environment(actiondimension, statusdimension, rewardimension, EnvType.minpath)
            edges = envspecs[EnvSpecs.edges]
            prize = envspecs[EnvSpecs.prize]
            penalty = envspecs[EnvSpecs.penalty]
            costs = envspecs[EnvSpecs.costs]
            finalpoint = envspecs[EnvSpecs.finalpoint]
            startingpoint = envspecs[EnvSpecs.startingpoint]
            self.env.set_linear_reward(costs)
            self.env.set_minpath(finalpoint, edges, prize, penalty)

            a0 = (startingpoint, startingpoint)
            st0 = {edge: 0.0 for edge in edges}
            neighbors = self.env.neighbors0(a0)

            return st0, neighbors
        else:
            raise Exception("ERROR Environment not yet implemented")

    def instance(self, st):
        if self._type == EnvType.minpath:

            return list(st.values()) + self.env.costlist()
        else:
            raise Exception("ERROR environment not supported.")

    def evaluate(self, vals, neighbors):
        neighborsp = self.env.decode( neighbors )
        #print("neighborspppp", neighborsp, vals)
        effective_dict = { neigh: vals[neighborsp[neigh]] for neigh in neighbors}
        return effective_dict
    def maxaction(self, effective_dict):
        maxval = -1e20
        action = (0,0)

        for elem in effective_dict:
            if effective_dict[elem] > maxval:
                maxval = effective_dict[elem]
                action = elem
        return action


    def solve(self, nepisodes = 100, noptsteps=1, randomness0 = 1, maximumiter=100, batchsize=5, display=(False,0), steps=0):

        st0, neighbors0 = self._buildenvironment(self._envspecs)
        self.env.set_multiple_respose()
        stats = []
        alpha = 0

        dispfreq = display[1]
        display = display[0]


        for episode in range(nepisodes):
            stat = {"counts": 0,
                    "final_objective": 0,
                    "cumulative_reward": []}
            cumRM = [0]
            neighbors = neighbors0.copy()
            st = st0.copy()
            RM = 0 #cumulative reward
            randomness = randomness0 / ((episode + alpha) * 0.05 + 1)
            if display and episode % dispfreq == 0:
                print("Ep."+str(episode)+")")
                print(randomness)

            feasible = False
            sol = []



            for t in range(maximumiter):
                #if display:
                #    print("      iter.",t,"  RM.", RM)
                answer = self._model.coremdl(torch.as_tensor(self.instance(st)))
                qvalues = self.evaluate(answer,  neighbors)

                at1 = (0,0)
                if random.random() <= randomness:
                    at1 = neighbors[random.randint(0, len(neighbors)-1)]
                else:
                    at1 = self.maxaction(qvalues)
                sol.append(at1)
                # if display:
                #     print("           move", at1)
                st1, (rt, rs), final, neighbors, feasible = self.env.response(st, at1)


                rt0 = rt
                if final:
                    RM += rt0 - self.env.prize()
                else:
                    qnext = max(
                        self.evaluate(self._model.coremdl(torch.as_tensor(self.instance(st1))), neighbors).values())
                    RM += rt0
                cumRM.append(RM)
                finalp = False
                qnextp = qnext


                if not final:
                    sp = st1.copy()
                    neighborsp = neighbors.copy()
                    for f in range(steps):
                        qvaluesp = self.evaluate(self._model.coremdl(torch.as_tensor(self.instance(sp))),  neighborsp)
                        ap1 = self.maxaction(qvaluesp)
                        sp1, ( rpbase, rsbase), finalp, neighborsp, feasp = self.env.response(sp, ap1)
                        rt += rpbase
                        if finalp:
                            break
                        else:
                            qnextp = max(self.evaluate(self._model.coremdl(torch.as_tensor(self.instance(sp1))),
                                                       neighborsp).values())

                        sp = sp1.copy()
                if finalp or final:
                    rs[at1] = -2*(RM - rt0 + rt)
                    self._data.add(self.instance(st), list(rs.values()))
                else:
                    rs[at1] = rt + qnext
                    self._data.add(self.instance(st), list(rs.values()))


                if final or self._stop_function(0): #todo correct for stop functions
                    break
                else:
                    xx, yy = self._data.get_batch(batchsize)
                    xx = torch.as_tensor(xx)
                    yy = torch.as_tensor(yy)
                    self._model.long_update(xx, yy, noptsteps)
                    st = st1
            if not final and feasible:
                feasible = False

            stat["cumulative_reward"] = cumRM
            stat["counts"] = len(cumRM)
            stat["final_objective"] = RM
            stat["is_final"] = 1 if feasible else 0
            stat["solution"] = sol
            stats.append(stat)
        return stats
