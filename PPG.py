from model import PPG_Model
from modules2 import Actor,Critic
from data_manager import PPG_data_manager
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
        self.Buffer = PPG_data_manager(stacklenght=stacklenght)
        self.phases = phases

        self.ACmodel = PPG_Model(specsActor=self.specsActor,
            specsCritic=self.specsCritic)

    def buildenvironment(self,envspecs):
        self.env = tsp(envspecs)
        self.rollout_lenght = self.env.nnodes

    def perform_rollouts(self,policy,s0,m0):
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
            action_probs = policy([torch.as_tensor(mat_inc_st)],[mt])[0]
            #print(action_probs)
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
        rew, pol_probs, states,masks, cum_rew, action_list, actions_ind_list = self.perform_rollouts(cp, s0, m0)
        cr = cum_rew.copy()
        cr.reverse()
        V_target_list = cr
        V_pred_list = [crr([torch.as_tensor(elem)])[0] for elem in states]
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




    def PPG_phase(self):
        #self.buildenvironment(envspecs)
        s0 = self.env.initial_state()
        m0 = self.env.initial_mask(s0)
        memory = self.Buffer
        for it in range(self.policy_iterations):
            print('Policy iteration n.',it+1)
            self.ACmodel.set_optim()
            self.ACmodel.set_loss()
            #current_policy = copy.deepcopy(self.ACmodel.actor)
            dataset = []
            for j in range(self.env.repetitions):
                B = self.policy_rollout(s0,m0,j,copy.deepcopy(self.ACmodel.actor),copy.deepcopy(self.ACmodel.critic))
                dataset.append(B)
                self.Buffer.add(B)
            for epoch in range(self.E_policy):
                print('Epoch n. ',epoch+1)
                Ls = []
                Lv = []
                for rep in range(self.env.repetitions):
                    #print('rep. n. ',rep+1)
                    data_rep = dataset[rep]
                    values_target = torch.as_tensor(data_rep['values'])
                    adv = torch.as_tensor(data_rep['advantages'])
                    values_pred = torch.empty(size=(self.rollout_lenght,))
                    masks_list = data_rep['masks']
                    states_list = data_rep['state']
                    old_action_probs = data_rep['actions_probs']
                    action_probs = self.ACmodel.actor(states_list,masks_list)
                    action_inds = data_rep['actions_ind']
                    loss_actor = self.ACmodel.update_actor(old_action_probs,action_probs,action_inds,adv)
                    Ls.append(loss_actor)
                    #print('Loss actor: ',loss_actor)
                    values = self.ACmodel.critic(states_list)
                    for k in range(len(values)):
                        values_pred[k] = values[k]

                    loss_critic = self.ACmodel.update_critic(values_pred,values_target)
                    #print('Loss critic: ',loss_critic)
                    Lv.append(loss_critic)

                print('Loss media actor: ',sum(Ls)/len(Ls))
                print('Loss media critic: ', sum(Lv)/len(Lv))

        for aux_epoch in range(self.E_aux):
            print('Auxiliary epoch n. ',aux_epoch+1)
            LJ, LA = [], []
            #L_ind = [_ for _ in range(self.Buffer.get_lenght())]
            #count = 0
            for ii in range(int(2*self.Buffer.get_lenght()/self.batchsize)):
                oldprobs, values, actinds, adv, states, mask_list_aux = self.Buffer.get_batch(self.batchsize)
                v_pred = self.ACmodel.critic(states)
                probs_aux = self.ACmodel.actor(states,mask_list_aux)
                L_value, L_joint = self.ACmodel.Loss_joint(v_pred,torch.as_tensor(values),oldprobs,probs_aux,self.batchsize)
                self.ACmodel.opt_actor.zero_grad()
                L_joint.backward(retain_graph=True)
                self.ACmodel.opt_actor.step()
                self.ACmodel.opt_critic.zero_grad()
                L_value.backward()
                self.ACmodel.opt_critic.step()
                LJ.append(L_joint)
                LA.append(L_value)
                #count += self.batchsize
            print('Loss Joint: {:4f},  Loss Aux: {:4f}'.format(sum(LJ)/len(LJ),sum(LA)/len(LA)))



        return self.ACmodel.actor,self.ACmodel.critic,self.Buffer

    def PPG_algo(self,envspecs):
        self.buildenvironment(envspecs)
        s0 = self.env.initial_state()
        m0 = self.env.initial_mask(s0)
        for phase in range(self.phases):
            actor,critic,current_buffer = self.PPG_phase()
            self.Buffer.restart()
        return self.ACmodel.actor,self.ACmodel.critic

    def Test_Policy(self,actor,critc):
        s0 = self.env.initial_state()
        m0 = self.env.initial_mask(s0)
        stats = []
        for rep in range(self.env.repetitions):
            B = self.policy_rollout(s0,m0,rep,actor,critc)
            stats.append(B)
        return stats


































