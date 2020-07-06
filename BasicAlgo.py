from model import Model
from data_manager import data_manager
from environment import environment, min_path, EnvSpecs, EnvType
from randomness import randomness, ExplorationSensitivity
import torch
import random
import copy
import time

class basicalgo():
    def __init__(self, environment_specs,
                 D_in,
                 modelspecs,
                 criterion = "mse",
                 optimizer = "sgd",
                 optspecs = None,
                 memorylength = None,
                 memorypath = None,
                 seed = None,
                 stop_function = None): #todo implement stopping functions
        if optspecs is None:
            optspecs = {"lr": 1e-4}
        self._model = Model(D_in, modelspecs, seed)
        self._model.set_loss(criterion)
        self._model.set_optimizer(name=optimizer, options=optspecs)
        self._data  = data_manager(stacklength=memorylength, seed=seed)
        if memorypath != None:
            self._data.memoryrecoverCSV(memorypath)
        self._stop_function = stop_function
        if stop_function == None:
            self._stop_function = basicalgo._no_stop
        self._envspecs = environment_specs
        self._buildenvironment(self._envspecs)

    def _no_stop(elem):
        return False

    def _buildenvironment(self, envspecs):
        if envspecs[EnvSpecs.type] == EnvType.min_path:
            self.env = min_path(envspecs)
        self._costs = self.env.linear_costs()

    def solve(self, nepisodes=100, noptsteps=1, randomness=randomness(1,rule = ExplorationSensitivity.linear_threshold, threshold = 0.02), maximumiter=1000, batchsize=5, display=(False, 0), steps=0, backcopy=0):
        st0 = self.env.initial_state()
        mask0 = self.env.initial_mask(st0)
        stats = []
        ccc = 0

        dispfreq = display[1]
        display = display[0]
        coremdl0 = copy.deepcopy(self._model.coremdl)

        for episode in range(nepisodes):
            if episode % len(self._costs) == 0:
                indices = random.sample(range(0, len(self._costs)), len(self._costs))
            rep = indices[episode % len(self._costs)]
            self.env.set_linear_reward(self._costs[rep])
            stat = {"counts": 0,
                    "final_objective": 0,
                    "cumulative_reward": [],
                    "rep":rep}
            cumRM = [0]
            mask = copy.deepcopy(mask0)
            st = copy.deepcopy(st0)
            RM = 0  # cumulative reward
            eps = randomness.next(episode)
            if display and episode % dispfreq == 0:
                print("Ep." + str(episode) + ")")
                print(eps)
            feasible = False
            sol = []
            insts = self.env.instances(st, mask)
            last_states = self.env.last_states()
            for t in range(maximumiter):
                if random.random() <= eps:
                    ract = random.randint(0, len(mask) - 1)
                    at1 = mask[ract]
                else:
                    qvalues = {m: self._model.coremdl(torch.as_tensor(insts[m])) for m in mask}
                    at1 = max(qvalues, key=qvalues.get)
                sol.append(at1)
                st1, rt, final, mask, feasible, inst = self.env.output(st, at1, last_states, insts)  # save computations by storing the last instances computed
                # st1, (rt, rs), final, neighbors, feasible = self.env.response(st, at1)

                rt0 = rt
                if final:
                    RM += rt0 - self.env.prize()
                else:
                    insts = self.env.instances(st1, mask)
                    last_states = self.env.last_states()
                    qvalues = {m: coremdl0(torch.as_tensor(insts[m])) for m in mask}
                    atnext = max(qvalues, key=qvalues.get)
                    qnext = qvalues[atnext]
                    RM += rt0
                cumRM.append(RM)
                finalp = False
                qnextp = qnext
                feasiblep = True

                if not final:
                    sp = st1.copy()
                    ap = max(qvalues, key=qvalues.get)
                    qnextp = qvalues[ap]
                    for f in range(steps):
                        sp, rp, finalp, maskp, feasiblep, instp = self.env.output(sp, ap)  # save computations by storing the last instances computed
                        rt += rp
                        if finalp:
                            break
                        else:
                            instsp = self.env.instances(sp, maskp)
                            qvaluesp = {m: coremdl0(torch.as_tensor(instsp[m])) for m in maskp}
                            ap = max(qvaluesp, key=qvaluesp.get)
                            qnextp = qvaluesp[ap]
                if finalp or final:
                    self._data.add(inst, self.env.prize() + RM - rt0 + rt)
                else:
                    self._data.add(inst, rt + (qnextp if feasible and feasiblep else self.env.penalty()))

                xx, yy = self._data.get_batch(batchsize)
                xx = torch.as_tensor(xx)
                yy = torch.as_tensor(yy)
                self._model.long_update(xx, yy, noptsteps)
                ccc += 1

                if ccc % (backcopy + 1) == 0:
                    coremdl0 = copy.deepcopy(self._model.coremdl)

                if final or self._stop_function(0):  # todo correct for stop functions
                    break
                else:
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

    def test(self, testcosts, maximumiter=1000):
        st0 = self.env.initial_state()
        mask0 = self.env.initial_mask(st0)
        stats = []
        nepisodes = len(testcosts)

        for episode in range(nepisodes):
            start = time.time()
            self.env.set_linear_reward(testcosts[episode])
            stat = {"counts": 0,
                    "final_objective": 0,
                    "cumulative_reward": [],
                    "rep":episode}
            cumRM = [0]
            mask = copy.deepcopy(mask0)
            st = copy.deepcopy(st0)
            RM = 0
            print("Inst."+str(episode)+")")
            feasible = False
            sol = []
            insts = self.env.instances(st, mask)
            qvalues = {m: self._model.coremdl(torch.as_tensor(insts[m])) for m in mask}
            for t in range(maximumiter):
                at1 = max(qvalues, key = qvalues.get)
                sol.append(at1)
                st1, rt, final, mask, feasible, inst = self.env.output(st, at1)
                rt0 = rt
                if final:
                    RM += rt0
                else:
                    insts= self.env.instances(st1, mask)
                    qvalues = {m: self._model.coremdl(torch.as_tensor(insts[m])) for m in mask}
                    RM += rt0
                cumRM.append(RM)
                if final or self._stop_function(0): #todo correct for stop functions
                    break
                else:
                    st = st1
            if not final and feasible:
                feasible = False

            stat["cumulative_reward"] = cumRM
            stat["counts"] = len(cumRM)
            stat["final_objective"] = RM
            stat["is_final"] = 1 if feasible else 0
            stat["solution"] = sol
            stat["time"] = time.time() - start
            stats.append(stat)
        return stats







