from model import PPG_Model
from modules2 import Actor,Critic
from data_manager import data_manager
from Environment_GH import environment, EnvSpecs, EnvType, tsp
import copy
import random
import torch



class ppg():

    def __init__(self,phases,
        policy_iterations,
        specsActor,
        specsCritic,
        E_policy,
        E_value,
        E_aux,
        stacklenght,
        seed,
        batchsize=1):

        self.policy_iterations = policy_iterations
        self.E_policy = E_policy
        self.E_value = E_value
        self.E_aux = E_aux
        self.batchsize = batchsize
        self.specsActor = specsActor
        self.specsCritic = specsCritic
        self.Buffer = data_manager(stacklength=stacklenght,seed=seed)

        self.ACmodel = PPG_Model(specsActor=self.specsActor,
            specsCritic=self.specsCritic)

    def buildenvironment(self,envspecs):
        self.env = tsp(envspecs)
        self.rollout_lenght = self.env.nnodes

    def perform_rollouts(self,max_ep,policy,s0,m0):
        rewards = []
        states = []
        pred_values = []
        st,mt = s0,m0
        mat_inc_st = self.env.incidence_matrix
        cum_rew = [0]
        states = [mat_inc_st]
        mask_list = [mt]
        pol_probs = []
        rew = []
        action_list = []
        actions_ind_list = []
        for j in range(self.rollout_lenght):
            f_insts = self.env.instances(st, mt)
            ns = mat_inc_st.shape[0]
            action_probs = policy(torch.as_tensor(mat_inc_st).view(-1, 1, ns, ns),mt)
            pol_probs.append(action_probs)
            dist_act_prob = torch.distributions.Categorical(action_probs)
            act_ind = dist_act_prob.sample().item()
            actions_ind_list.append(act_ind)
            act = mt[act_ind]
            action_list.append(act)
            st, rt, final, mt, feasible, instt = self.env.output(st, act)
            mat_inc_st = instt[4]
            states.append(mat_inc_st)
            rew.append(rt)
            mask_list.append(mt)
            cum_rew += [rt + cum_rew[-1]]
            if final:
                break
        return rew,pol_probs,states,mask_list,cum_rew,action_list, actions_ind_list

    def policy_rollout(self,s0,m0,rep,cp,crr):
        B = {'state': [], 'actions_probs': [], 'advantages':[],'values':[],'masks':[],'actions':[],'actions_ind':[]}
        self.env.set_instance(rep)
        rew, pol_probs, states,masks, cum_rew, action_list, actions_ind_list = self.perform_rollouts(self.env.repetitions - 1, cp, s0, m0)
        cr = cum_rew.copy()
        cr.reverse()
        V_target_list = cr
        V_pred_list = [crr(torch.as_tensor(elem).view(-1, 1, elem.shape[0], elem.shape[0])).item() for elem in states]
        adv = []
        for ik in range(len(states) - 1):
            ai = V_pred_list[ik + 1] - V_pred_list[ik] + rew[ik]
            adv.append(ai)
        adv.append(0)
        v_pred = torch.Tensor(V_pred_list)
        v_targ = torch.Tensor(V_target_list)
        advantages = torch.Tensor(adv)
        for k in range(len(states)-1):
            B['state'].append(states[k])
            B['actions_probs'].append(pol_probs[k])
            B['advantages'].append(adv[k])
            B['values'].append(V_target_list[k])
            B['masks'].append(masks[k])
            B['actions'].append(action_list[k])
            B['actions_ind'].append(actions_ind_list[k])
        return B




    def PPG_algo(self,envspecs):
        self.buildenvironment(envspecs)
        s0 = self.env.initial_state()
        m0 = self.env.initial_mask(s0)
        for it in range(self.policy_iterations):
            print('Policy iteration n.',it+1)
            self.ACmodel.set_optim()
            self.ACmodel.set_loss()
            #current_policy = copy.deepcopy(self.ACmodel.actor)
            dataset = []
            for j in range(self.env.repetitions-1):
                B = self.policy_rollout(s0,m0,j,copy.deepcopy(self.ACmodel.actor),copy.deepcopy(self.ACmodel.critic))
                dataset.append(B)
            for epoch in range(self.E_policy):
                print('Epoch n. ',epoch+1)
                Ls = []
                Lv = []
                #current_critic = copy.deepcopy(self.ACmodel.critic)
                #current_policy = copy.deepcopy(self.ACmodel.actor)
                for rep in range(self.env.repetitions-1):
                    print('rep. n. ',rep+1)
                    data_rep = dataset[rep]
                    old_action_probs = None
                    action_probs = None
                    #values_pred = torch.nn.ModuleList(self.rollout_lenght)
                    #values_pred.requires_grad = True
                    values_target = torch.as_tensor(data_rep['values'])
                    adv = torch.as_tensor(data_rep['advantages'])
                    values_pred = torch.empty(size=(self.rollout_lenght,))
                    entropy = None
                    for timestep in range(self.rollout_lenght): #[0.1,0.7,0.2]
                        state_t = data_rep['state'][timestep]
                        mask_t = data_rep['masks'][timestep]
                        ind_act_t = data_rep['actions_ind'][timestep]
                        all_probs = data_rep['actions_probs'][timestep]
                        prob_at = all_probs[ind_act_t]
                        ns = state_t.shape[0]
                        all_probs_pred = self.ACmodel.actor(torch.as_tensor(state_t).view(-1,1,ns,ns),mask_t)
                        prob_at_pred = all_probs_pred[ind_act_t]
                        dist = torch.distributions.Categorical(all_probs_pred)
                        entropy_t = dist.entropy()
                        #entropy_t = torch.sum(all_probs_pred)
                        value_t = self.ACmodel.critic(torch.as_tensor(state_t).view(-1,1,ns,ns))
                        #print('value_t= ',value_t)
                        if old_action_probs is None:
                            old_action_probs = prob_at.view(1)
                            action_probs = prob_at_pred.view(1)
                            entropy = entropy_t.view(1)
                            values_pred[timestep] = value_t
                        else:
                            old_action_probs = torch.cat((old_action_probs,prob_at.view(1)))
                            action_probs = torch.cat((action_probs,prob_at_pred.view(1)))
                            entropy = torch.cat((entropy,entropy_t.view(1)))
                            values_pred[timestep] = value_t

                    #print(action_probs)
                    #print(old_action_probs)
                    ratios = action_probs/old_action_probs
                    #print(ratios)
                    print('adv',adv)
                    surr1 = ratios*adv
                    surr2 = ratios.clamp(0.8,1.2)*adv
                    obj = torch.min(surr1,surr2) + entropy
                    #print('surr1',surr1)
                    #print('surr2', surr2)
                    #print('entropy',entropy)
                    loss = -obj.mean()
                    print('loss',loss)
                    self.ACmodel.opt_actor.zero_grad()
                    loss.backward(retain_graph=True)
                    self.ACmodel.opt_actor.step()
                    #print('actor aggiornato con policy old mantenuta')
                    Ls.append(loss.item())

                    #lvv = torch.nn.MSELoss()
                    #print(lvv)
                    #torch.autograd.set_detect_anomaly(True)
                    #loss_value = lvv(values_pred,values_target)
                    #print(values_pred)
                    #print(values_target)
                    #self.ACmodel.opt_critic.zero_grad()
                    #loss_value = lvv(values_pred, values_target)
                    #print(loss_value)
                    #loss_value.backward()
                    #self.ACmodel.opt_critic.step()
                    #print('critic aggiornato')
                    #Lv.append(loss_value)





                print('Loss Media: ', sum(Ls)/len(Ls))
                print('Loss Media Value: ', sum(Lv) / len(Lv))
                print('\n')


            current_policy = copy.deepcopy(self.ACmodel.actor)
            #print('!Aggiornamento della policy effettuato!')

        return 'gatto'



























