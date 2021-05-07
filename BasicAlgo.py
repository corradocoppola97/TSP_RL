from model import Model
from data_manager import data_manager
from Environment_GH import environment, min_path, bnb1, EnvSpecs, EnvType, job_shop_scheduling, js_LSTM, tsp
from randomness import randomness, ExplorationSensitivity
import torch
#torch.set_num_threads(4)
import random
import copy
import time

class basicalgo():
    def __init__(self, environment_specs,
                 D_in,
                 modelspecs,edges,nnodes,mod_layers,
                 criterion = "mse",
                 optimizer = "sgd",
                 optspecs = None,
                 scheduler = "multiplicative",
                 schedspecs = None,
                 memorylength = None,
                 memorypath = None,
                 seed = None,
                 stop_function = None): #todo implement stopping functions
        if optspecs is None:
            optspecs = {"lr": 1e-4}
        self._model = Model(D_in, modelspecs,edges,nnodes,mod_layers=mod_layers,seed=seed)
        #self._model = Model(D_in, modelspecs, LSTMflag = True, seed = seed)
        #self._model = Model(D_in, modelspecs, seed, environment_specs[EnvSpecs.edges], environment_specs[EnvSpecs.finalpoint]+1)
        self._model.set_loss(criterion)
        self._model.set_optimizer(name=optimizer, options=optspecs)
        self._model.set_scheduler(name=scheduler, options=schedspecs)
        self._data  = data_manager(stacklength=memorylength, seed=seed)
        if memorypath != None:
            self._data.memoryrecoverCSV(memorypath)
        self._stop_function = stop_function
        if stop_function == None:
            self._stop_function = basicalgo._no_stop
        self._envspecs = environment_specs
        self._buildenvironment(self._envspecs)
        self.nedges = len(edges)

    def _no_stop(elem):
        return False

    def _buildenvironment(self, envspecs):
        if envspecs[EnvSpecs.type] == EnvType.min_path:
            self.env = min_path(envspecs)
        if envspecs[EnvSpecs.type] == EnvType.bnb1:
            self.env = bnb1(envspecs)
        if envspecs[EnvSpecs.type] == EnvType.job_shop_scheduling:
            self.env = job_shop_scheduling(envspecs)
        if envspecs[EnvSpecs.type] == EnvType.js_LSTM:
            self.env = js_LSTM(envspecs)
        if envspecs[EnvSpecs.type] == EnvType.tsp:
            self.env = tsp(envspecs)

    def solve(self, repetitions, nepisodes=5000, noptsteps=1,
              randomness=randomness(1,rule = ExplorationSensitivity.linear_threshold, threshold = 0.02),
              maximumiter=1000, batchsize=1, display=(False, 0), steps=5, backcopy=0):
        st0 = self.env.initial_state()
        mask0 = self.env.initial_mask(st0)
        stats = []
        ccc = 0
        l_fob = []
        mov_avg = []
        control = 5
        th = 2000

        displayflag = display[0]
        dispfreq = display[1]
        displayloss1000 = display[2] if len(display) > 2 else False
        coremdl0 = copy.deepcopy(self._model.coremdl)

        for episode in range(nepisodes):
            print('Inizio episodio {} di {}'.format(episode+1,nepisodes))
            if episode % repetitions == 0:
                indices = random.sample(range(0, repetitions), repetitions)
                if episode > 0:
                    self._model.schedulerstep()
            rep = indices[episode % repetitions]
            self.env.set_instance(rep)
            stat = {"counts": 0,
                    "final_objective": 0,
                    "cumulative_reward": [],
                    "rep":rep}
            cumRM = [0]
            mask = copy.deepcopy(mask0)
            st = copy.deepcopy(st0)
            RM = 0  # cumulative reward
            eps = randomness.next(episode)
            feasible = False
            sol = []
            insts = self.env.instances(st, mask)
            last_states = copy.deepcopy(self.env.last_states())
            niter=0
            #print('Inizio ricerca azioni')
            #maximumiter = 10
            print('eps', eps)
            for t in range(maximumiter):
                #print('Iterazione {} di {}'.format(t+1,maximumiter))
                niter+=1
                if random.random() <= eps:
                    ract = random.randint(0, len(mask) - 1)
                    at1 = mask[ract]
                else:
                    qvalues = {m: coremdl0([insts[m]]) for m in mask}
                    #print('qvalues', qvalues)
                    at1 = max(qvalues, key=qvalues.get)
                sol.append(at1)
                #print("atn", at1)
                #print('M1')
                st, rt, final, mask, feasible, inst = self.env.output(st, at1, last_states, insts)
                #print('M2')

                rt0 = rt
                RM += rt0
                cumRM.append(RM)
                finalp = False
                feasiblep = feasible
                qnextp = 0
                if not final and feasible:
                    #print('M3')
                    insts = self.env.instances(st, mask)
                    #print('M4')
                    last_states = self.env.last_states()
                    #print('M5')
                    #print('coremdl0 = ', coremdl0)
                    qvalues = {mm: coremdl0([insts[mm]]) for mm in mask}
                    #print('qvaules',qvalues.values())
                    #print('M6')
                    #print('M7')

                    atnext = max(qvalues, key=qvalues.get)
                    #print("atn",atnext)
                    qnext = qvalues[atnext].detach().numpy()

                    sp = st
                    ap = atnext
                    qnextp = qnext
                    #print('Inizio step')
                    for f in range(steps):
                        #print('Step {} di {}'.format(f+1,steps))
                        #print('M8')
                        sp, rp, finalp, maskp, feasiblep, instp = self.env.output(sp, ap)
                        #print('M9')
                        rt += rp
                        if finalp or not feasiblep:
                            break
                        else:
                            instsp = self.env.instances(sp, maskp)
                            qvaluesp = {mp: coremdl0([instsp[mp]]) for mp in maskp}
                            ap = max(qvaluesp, key=qvaluesp.get)
                            qnextp = qvaluesp[ap].detach().numpy()
                if finalp or final:
                    #print('aaaa',self.env.prize() + RM - rt0 + rt)
                    matinc_st = torch.as_tensor(inst[3])
                    at_one_hot = torch.as_tensor(inst[5])
                    matinc_st1 = torch.as_tensor(inst[4])
                    inst_s = [matinc_st, at_one_hot, matinc_st1]
                    #print('ris_fin',self.env.prize() + RM - rt0 + rt)
                    self._data.add(inst_s, self.env.prize() + RM - rt0 + rt)
                else:
                    matinc_st =  torch.as_tensor(inst[3])
                    at_one_hot = torch.as_tensor(inst[5])
                    matinc_st1 = torch.as_tensor(inst[4])
                    inst_s = [matinc_st,at_one_hot,matinc_st1]
                    self._data.add(inst_s, rt + (qnextp if feasible and feasiblep else self.env.penalty()))

                xx, yy = self._data.get_batch(batchsize)
                #print('xx: ', xx)
                #print('yy: ', yy)
                #print('len_xx', len(xx))
                #xx = xx[0]

                #print('yy',yy)
                #xx = torch.tensor(xx)
                yy = torch.as_tensor(yy).reshape((len(yy),1))
                self._model.long_update(xx, yy, noptsteps,bsize= batchsize)
                #print('L1', self._model.coremdl(xx))
                #print('L2', yy)
                #print('Loss: ',self._model.criterion(self._model.coremdl(xx),yy).item())
                ccc += 1

                if ccc % (backcopy + 1) == 0:
                    coremdl0 = copy.deepcopy(self._model.coremdl)

                if final or self._stop_function(0) or (not final and not feasible):  # todo correct for stop functions
                    break
            # if not final and feasible:
            #     feasible = False
            if displayflag and displayloss1000 and episode % dispfreq == 0:
                xx, yy = self._data.get_batch(1000, last=0)
                xx = torch.as_tensor(xx)
                yy = torch.as_tensor(yy).reshape((len(yy),1))
                print("Loss on last 1000:",self._model.criterion(self._model.coremdl(xx),yy).item())
            if episode==15:
                stat["Loss_in"]=(self._model.criterion(self._model.coremdl(xx),yy).item())
                print(stat["Loss_in"])
            if episode==nepisodes-1:
                stat["Loss_fin"]=(self._model.criterion(self._model.coremdl(xx),yy).item())
                print(stat["Loss_fin"])
            stat["cumulative_reward"] = cumRM
            stat["counts"] = len(cumRM)
            stat["final_objective"] = RM
            #print(RM)
            stat["is_final"] = 1 #if feasible else 0
            stat["solution"] = sol
            stats.append(stat)
            if stat["is_final"] == 1:
                print(stat['final_objective'])
                l_fob.append(stat['final_objective'])
                if len(l_fob)>=50:
                    lll = l_fob[len(l_fob)-50:]
                    lm = sum(lll)/50
                    mov_avg.append(lm)
                    print('mov_avg:',lm)



        return stats

    def test(self, testcosts, maximumiter=1000):
        st0 = self.env.initial_state()
        mask0 = self.env.initial_mask(st0)
        stats = []
        nepisodes = len(testcosts)

        for episode in range(nepisodes):
            start = time.time()
            self.env.set_instance(episode, testcosts)
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
            qvalues = {m: self._model.coremdl([insts[m]]) for m in mask}
            for t in range(maximumiter):
                at1 = max(qvalues, key = qvalues.get)
                sol.append(at1)
                st, rt, final, mask, feasible, inst = self.env.output(st, at1)
                rt0 = rt
                if final:
                    RM += rt0
                else:
                    insts= self.env.instances(st, mask)
                    qvalues = {m: self._model.coremdl([insts[m]]) for m in mask}
                    RM += rt0
                cumRM.append(RM)
                if final or self._stop_function(0): #todo correct for stop functions
                    break
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







