from model import Model
from data_manager import data_manager
from environment import environment, EnvType, RewType
from enum import Enum
import torch
import random
import math
import time
import numpy as np
import copy



class thor():
    def __init__(self, environment_specs,
                 D_in,
                 modelspecs,
                 criterion = "mse",
                 optimizer = "SGD",
                 optspecs = {"lr":1e-4},
                 memorylength = None,
                 memorypath = None,
                 seed = None,
                 stop_function = None,
                 repetitions = None): #todo implement stopping functions
        #torch.set_num_threads(1)
        self._model = Model(D_in, modelspecs, seed)
        print("D_in",D_in)
        print(modelspecs)
        self._model.set_loss(criterion)
        self._model.set_optimizer(name=optimizer, options=optspecs)
        self._data  = data_manager(stacklength=memorylength, seed=seed)
        if memorypath != None:
            self._data.memoryrecoverCSV(memorypath)
        self._stop_function = stop_function
        if stop_function == None:
            self._stop_function = thor._no_stop
        self._envspecs = environment_specs
        self._repetitions = repetitions
        if repetitions is None:
            self._repetitions = 1


    def _no_stop(elem):
        return False

    def _buildenvironment(self, envspecs):
        if envspecs[EnvSpecs.type] == EnvType.minpath:
            self._type = EnvType.minpath
            actiondimension = envspecs[EnvSpecs.actiondimension]
            statusdimension = envspecs[EnvSpecs.statusdimension]
            rewardimension = envspecs[EnvSpecs.rewardimension]
            self.env = environment(actiondimension, statusdimension, rewardimension, EnvType.minpath)
            edges = envspecs[EnvSpecs.edges]
            prize = envspecs[EnvSpecs.prize]
            penalty = envspecs[EnvSpecs.penalty]
            self._costs = envspecs[EnvSpecs.costs]
            finalpoint = envspecs[EnvSpecs.finalpoint]
            startingpoint = envspecs[EnvSpecs.startingpoint]
            self._nullAlternative = envspecs.get(EnvSpecs.nullalternative)
            self._nullCheck = self._nullAlternative is not None
            # self.env.set_linear_reward(costs)
            self.env.set_minpath(finalpoint, edges, prize, penalty)

            a0 = (startingpoint, startingpoint)
            st0 = {edge: 0.0 for edge in edges}
            neighbors = self.env.neighbors0(a0)

            return st0, neighbors
        elif envspecs[EnvSpecs.type] == EnvType.min_path_compressed:
            self._type = EnvType.min_path_compressed
            actiondimension =  envspecs[EnvSpecs.actiondimension]
            statusdimension =  envspecs[EnvSpecs.statusdimension]
            rewardimension  =  envspecs[EnvSpecs.rewardimension]
            self.env = environment(actiondimension, statusdimension, rewardimension, self._type)
            edges = envspecs[EnvSpecs.edges]
            prize = envspecs[EnvSpecs.prize]
            penalty = envspecs[EnvSpecs.penalty]
            self._costs = envspecs[EnvSpecs.costs]
            finalpoint = envspecs[EnvSpecs.finalpoint]
            startingpoint = envspecs[EnvSpecs.startingpoint]
            self._nullAlternative = envspecs.get(EnvSpecs.nullalternative)
            self._nullCheck = self._nullAlternative is not None
            #self.env.set_linear_reward(costs)
            self.env.set_minpath_compressed(finalpoint, edges, prize, penalty)

            a0 = (startingpoint, startingpoint)
            st0 = {edge: 0.0 for edge in edges}
            neighbors = self.env.neighbors0(a0)

            return st0, neighbors
        else:
            raise Exception("ERROR Environment not yet implemented")

    def instance(self, st, action, st1 = None):
        if self._type == EnvType.minpath:
            if st1 is None:
                # print("st",st)
                st1, dummy1, dummy2, dummy3, dummy4 = self.env.response(st, action)
            return list(st.values()) + list(self.env.encode(action).values()) + list(st1.values()) + self.env.costlist()
        elif self._type == EnvType.min_path_compressed:
            if st1 is None:
                st1, dummy1, dummy2, dummy3, dummy4 = self.env.response(st, action)
            return list(st.values()) + list(st1.values())
        else:
            raise Exception("ERROR environment not supported.")

    def solve(self, nepisodes = 100, noptsteps=1, randomness0 = 1, maximumiter=100, batchsize=5, display=(False,0), steps=3):

        st0, neighbors0 = self._buildenvironment(self._envspecs)
        stats = []
        alpha = 0

        dispfreq = display[1]
        display = display[0]
        randomness = randomness0

        coremod0 = copy.deepcopy(self._model.coremdl)


        for episode in range(nepisodes):
            rep = random.randint(a=0, b=self._repetitions-1)
            self.env.set_linear_reward(self._costs[rep])
            stat = {"counts": 0,
                    "final_objective": 0,
                    "cumulative_reward": [],
                    "rep":rep}
            cumRM = [0]
            neighbors = neighbors0.copy()
            st = st0.copy()
            RM = 0 #cumulative reward
            if randomness >= 0.02:
                randomness = randomness0 / ((episode + alpha) * 0.01 + 1)
            else:
                if episode < 15000:
                    randomness = 0.02
                else:
                    randomness = 0.002
            if display and episode % dispfreq == 0:
                print("Ep."+str(episode)+")")
                print(randomness)







            feasible = False
            sol = []
            appliedmaximumiter = max(math.ceil(episode/15)+2,maximumiter)


            for t in range(appliedmaximumiter):
                #if display:
                #    print("      iter.",t,"  RM.", RM)
                qvalues = [self._model.coremdl(torch.as_tensor(self.instance(st, neigh))) for neigh in neighbors]

                at1 = (0,0)
                if random.random() <= randomness:
                    at1 = neighbors[random.randint(0, len(neighbors)-1)]
                else:
                    at1 = self.env.action(qvalues, neighbors)
                if self._nullCheck and at1 == self._nullAlternative :
                    self._data.add(self.instance(st, at1, st), [0])
                    break

                sol.append(at1)
                # if display:
                #     print("           move", at1)
                st1, rt, final, neighbors, feasible = self.env.response(st, at1)
                qnext = coremod0(torch.as_tensor(self.instance(st1, at1)))

                rt0 = rt
                if final:
                    RM += rt0 - self.env.prize()
                else:
                    RM += rt0
                cumRM.append(RM)
                finalp = False
                qnextp = qnext


                if not final:
                    sp = st1.copy()
                    neighborsp = neighbors.copy()
                    for f in range(steps):
                        qvaluesp = [self._model.coremdl(torch.as_tensor(self.instance(sp, neigh))) for neigh in
                                    neighborsp]

                        ap1 = self.env.action(qvaluesp, neighborsp)
                        if self._nullCheck and ap1 == self._nullAlternative:
                            finalp = False
                            qnextp = 0
                            # rt = 0
                            break
                        sp1, rpbase, finalp, neighborsp, feasp = self.env.response(sp, ap1)
                        qnextp = coremod0(torch.as_tensor(self.instance(sp1, ap1)))
                        rt += rpbase
                        if finalp:
                            break
                        sp = sp1.copy()

                ist = self.instance(st, at1, st1)
                if finalp or final:
                    self._data.add(ist, [10000 + (RM -rt0 + rt)])
                else:
                    self._data.add(ist, [rt + qnextp])


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

            if episode%30 == 0:
                coremod0 = copy.deepcopy(self._model.coremdl)
        return stats

    def test(self, testcosts,  maximumiter=100):
        st0, neighbors0 = self._buildenvironment(self._envspecs)
        stats = []
        nvariancereduction = 300
        variancereduction = []
        pastvariance = []
        npastvariance = 300
        randomness = 0
        nepisodes = len(testcosts)

        for episode in range(nepisodes):
            self.env.set_linear_reward(testcosts[episode])
            stat = {"counts": 0,
                    "final_objective": 0,
                    "cumulative_reward": [],
                    "test no": episode}
            cumRM = [0]
            start = time.time()
            neighbors = neighbors0.copy()
            st = st0.copy()
            RM = 0  # cumulative reward
            feasible = False
            sol = []
            for t in range(maximumiter):
                qvalues = [self._model.coremdl(torch.as_tensor(self.instance(st, neigh)))
                            for neigh in neighbors]
                at1 = self.env.action(qvalues, neighbors)
                sol.append(at1)
                st1, rt, final, neighbors, feasible = self.env.response(st, at1)
                rt0 = rt
                if final:
                    RM += rt0 - self.env.prize()
                else:
                    RM += rt0
                cumRM.append(RM)

                if final or self._stop_function(0):  # todo correct for stop functions
                    break
            if not final and feasible:
                feasible = False

            stat["cumulative_reward"] = cumRM
            stat["counts"] = len(cumRM)
            stat["final_objective"] = RM
            stat["time"] = time.time() - start

            stat["is_final"] = 1 if feasible else 0
            # print("RM", RM, " is it feas?", feasible)
            stat["solution"] = sol
            stats.append(stat)

            if len(variancereduction) < nvariancereduction:  # and feasible and final:
                variancereduction.append(abs(RM))
            elif len(variancereduction) == nvariancereduction:
                variancereduction.pop(0)
                variancereduction.append(abs(RM))

                sigma = np.std(variancereduction)
                if len(pastvariance) < npastvariance:
                    pastvariance.append(sigma)
                    print("lenpastvariance", len(pastvariance))
                else:
                    avgvariance = np.average(pastvariance)
                    varvar = np.std(pastvariance)
                    print("sigma", sigma, "avgvariance", avgvariance)
                    pastvariance.pop(0)
                    pastvariance.append(sigma)

        return stats

