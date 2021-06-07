from model import PPG_Model
from modules2 import Actor,Critic
from data_manager import PPG_data_manager
from Environment_GH import environment, EnvSpecs, EnvType, tsp
import copy
import random
import torch
from support import build_index,build_features,build_attr


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
        batchsize,
        exper,
        gamma,
        lam):

        self.policy_iterations = policy_iterations
        self.E_policy = E_policy
        self.E_value = E_value
        self.E_aux = E_aux
        self.batchsize = batchsize
        self.specsActor = specsActor
        self.specsCritic = specsCritic
        self.Buffer = PPG_data_manager(stacklenght=stacklenght)
        self.phases = phases
        self.experience_dataset_lenght = exper
        self.gamma = gamma
        self.lam = lam
        #self.policy_phase_buffer = PPG_data_manager(stacklenght=self.experience_dataset_lenght)


        self.ACmodel = PPG_Model(specsActor=self.specsActor,
            specsCritic=self.specsCritic)



    def buildenvironment(self,envspecs):
        self.env = tsp(envspecs)
        self.rollout_lenght = self.env.nnodes
        self.policy_phase_buffer = PPG_data_manager(stacklenght=self.Buffer.stacklenght) #PPG_data_manager(stacklenght=self.experience_dataset_lenght*self.rollout_lenght)

    def perform_rollouts(self,policy,s0,m0,threshold):
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
        edge_indexes_list = []
        features_list = []
        edge_attr_list = []
        for j in range(self.rollout_lenght):
            f_insts = self.env.instances(st, mt)
            ns = mat_inc_st.shape[0]
            #edge_ind = build_index(ns)
            #features = build_features(mat_inc_st,f_insts[mt[0]][0][1],self.env.incidence_matrix,self.env.current_depot)
            #edge_indexes_list.append(edge_ind)
            #features_list.append(features)
            action_probs = policy([torch.as_tensor(mat_inc_st)],[mt])[0]
            #edge_attr = build_attr(mat_inc_st)
            #edge_attr_list.append(edge_attr)
            #action_probs = policy([features],[edge_ind],[edge_attr])[0]
            pol_probs.append(action_probs)
            #print('act_probs', action_probs)
            #print('mask', mt)
            dist_act_prob = torch.distributions.Categorical(action_probs)
            eps = random.random()
            if eps<=threshold:
                act_ind = random.randint(0,len(action_probs)-1)
            else:
                act_ind = dist_act_prob.sample().item()
            #print('act_ind',act_ind)
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
        return rew,pol_probs,states,mask_list,cum_rew,action_list,actions_ind_list,edge_indexes_list,features_list,edge_attr_list

    def policy_rollout(self,s0,m0,rep,cp,crr,threshold):
        B = {'state': [], 'actions_probs': [], 'advantages':[],'values':[],'masks':[],'actions':[],'actions_ind':[]}
             #'edge_indexes_list':[],'features_list':[], 'edge_attr_list':[]}
        self.env.set_instance(rep)
        rew, pol_probs, states,masks, cum_rew, action_list, actions_ind_list, eil, fl, eal = self.perform_rollouts(cp, s0, m0,threshold)
        cr = cum_rew.copy()
        cr.reverse()
        V_target_list = cr
        V_pred_list = [crr([torch.as_tensor(elem)])[0] for elem in states]
        #V_pred_list = [crr([fl[k]],[eil[k]],[eal[k]])[0] for k in range(len(fl))]
        #print(V_pred_list)
        adv = []
        for ik in range(len(states) - 2):
            ai = V_pred_list[ik + 1] - V_pred_list[ik] + rew[ik]
            adv.append(ai)
        adv.append(0)

        #delta_list = []
        #for i in range(len(rew)):
            #delta = rew[i]+self.gamma*V_pred_list[i+1]-V_pred_list[i]
            #delta = delta.item()
            #delta_list.append(delta)

        #gae = 0
        #adv.append(gae)
        #for i in reversed(range(len(rew))):
            #gae = delta_list[i] + self.gamma*self.lam*adv[0]
            #adv.insert(0,gae)

        #print('adv',adv)
        v_pred = torch.Tensor(V_pred_list).detach()
        v_targ = torch.Tensor(V_target_list).detach()
        advantages = torch.Tensor(adv)
        for k in range(len(states)-1):
            B['state'].append(states[k])
            B['actions_probs'].append(pol_probs[k])
            B['advantages'].append(adv[k])
            B['values'].append(V_target_list[k])
            B['masks'].append(masks[k])
            B['actions'].append(action_list[k])
            B['actions_ind'].append(actions_ind_list[k])
            #B['edge_indexes_list'].append(eil[k])
            #B['features_list'].append(fl[k])
            #B['edge_attr_list'].append(eal[k])
        return B,cum_rew




    def PPG_phase(self,threshold):
        #self.buildenvironment(envspecs)
        phase_rew_stats = []
        s0 = self.env.initial_state()
        m0 = self.env.initial_mask(s0)
        memory = self.Buffer
        current_actor = copy.deepcopy(self.ACmodel.actor)
        current_critic = copy.deepcopy(self.ACmodel.critic)
        Loss_actor_stats, Loss_critic_stats = [], []
        for it in range(self.policy_iterations):
            #print('Policy iteration n.',it+1)
            self.ACmodel.set_optim()
            self.ACmodel.set_loss()
            it_rew_stats_list = []
            for rep in range(self.env.repetitions):
                rep_rew_stats_list = []
                for j in range(self.experience_dataset_lenght):
                    B, cum_rew = self.policy_rollout(s0,m0,rep,current_actor,current_critic,threshold)
                    self.Buffer.add(B)
                    self.policy_phase_buffer.add(B)
                    rep_rew_stats_list.append(cum_rew[-1])
                it_rew_stats_list.append(rep_rew_stats_list)
            #print(cum_rew_stats_list)
            #print('Average cum_reward: ',sum(cum_rew_stats_list)/len(cum_rew_stats_list))
            for epoch in range(self.E_policy):
                #print('Epoch n. ',epoch+1)
                Ls = [] #Loss actor
                Lv = [] #Loss critic
                nit = int(self.policy_phase_buffer.get_lenght()/self.batchsize)
                l_ind = [_ for _ in range(self.policy_phase_buffer.get_lenght())]
                for iteration in range(nit):
                    #if iteration%10 == 0:
                        #print('iteration {} di {}'.format(iteration+1,nit))
                    list_index = l_ind[iteration*self.batchsize:min(self.batchsize*(iteration+1),len(l_ind)-1)]
                    #print(list_index)
                    old_action_probs, values, action_inds, adv, states_list, masks_list = self.policy_phase_buffer.get_batch(self.batchsize)
                    values_target = torch.as_tensor(values)
                    adv = torch.as_tensor(adv)
                    values_pred = torch.empty(size=(self.batchsize,))
                    action_probs = self.ACmodel.actor(states_list,masks_list)
                    #action_probs = self.ACmodel.actor(feat_list,edg_inds,edge_ats)
                    loss_actor = self.ACmodel.update_actor(old_action_probs,action_probs,action_inds,adv)
                    Ls.append(loss_actor.item())
                    #print('Loss actor: ',loss_actor)
                    values = self.ACmodel.critic(states_list)
                    #values = self.ACmodel.critic(feat_list,edg_inds,edge_ats)
                    for k in range(len(values)):
                        values_pred[k] = values[k]

                    loss_critic = self.ACmodel.update_critic(values_pred,values_target)
                    #print('Loss critic: ',loss_critic)
                    Lv.append(loss_critic.item())

                #print('Loss media actor: ',sum(Ls)/len(Ls))
                #print('Loss media critic: ', sum(Lv)/len(Lv))
                Loss_critic_stats.append(Lv)
                Loss_actor_stats.append(Ls)

            self.policy_phase_buffer.restart()
            phase_rew_stats.append(it_rew_stats_list)
        Loss_aux_stats = []
        Loss_joint_stats = []
        for aux_epoch in range(self.E_aux):
            #print('Auxiliary epoch n. ',aux_epoch+1)
            LJ, LA = [], []
            #L_ind = [_ for _ in range(self.Buffer.get_lenght())]
            #count = 0
            nit_aux = int(self.Buffer.get_lenght()/self.batchsize)
            for ii in range(nit_aux):
                #if ii%10 == 0:
                    #print('Auxiliary iteration n. {} di {}'.format(ii+1,nit_aux))
                l_ind_aux = [_ for _ in range(ii*self.batchsize,min(self.batchsize*(ii+1),self.Buffer.get_lenght()))]
                #print(l_ind_aux)
                oldprobs, values, actinds, adv, states, mask_list_aux = self.Buffer.get_batch(self.batchsize,l_ind_aux)
                v_pred = self.ACmodel.critic(states)
                probs_aux = self.ACmodel.actor(states,mask_list_aux)
                #v_pred = self.ACmodel.critic(fl_aux,eil_aux,eal_aux)
                #probs_aux = self.ACmodel.actor(fl_aux,eil_aux,eal_aux)
                #print('probs_aux=',probs_aux)
                L_value, L_joint = self.ACmodel.Loss_joint(v_pred,torch.as_tensor(values),oldprobs,probs_aux,self.batchsize)
                self.ACmodel.opt_actor.zero_grad()
                L_joint.backward(retain_graph=True)
                self.ACmodel.opt_actor.step()
                self.ACmodel.opt_critic.zero_grad()
                L_value.backward()
                self.ACmodel.opt_critic.step()
                LJ.append(L_joint.item())
                LA.append(L_value.item())
                #count += self.batchsize
            #print('Loss Joint: {:4f},  Loss Aux: {:4f}'.format(sum(LJ)/len(LJ),sum(LA)/len(LA)))
            Loss_joint_stats.append(sum(LJ)/len(LJ))
            Loss_aux_stats.append(sum(LA)/len(LA))



        return self.ACmodel.actor,self.ACmodel.critic,self.Buffer,phase_rew_stats,Loss_actor_stats,Loss_critic_stats,Loss_joint_stats,Loss_aux_stats

    def PPG_algo(self,envspecs):
        self.buildenvironment(envspecs)
        s0 = self.env.initial_state()
        m0 = self.env.initial_mask(s0)
        stats_reward = []
        #stats_avg_reward = []
        stats_actor_loss = []
        stats_critic_loss = []
        stats_loss_joint = []
        stats_loss_aux = []
        threshold = 0.25
        for phase in range(self.phases):
            print('Inizio fase n. {} di {}'.format(phase+1,self.phases))
            actor,critic,current_buffer,rew_stats,Loss_actor_stats,Loss_critic_stats,Loss_joint_stats,Loss_aux_stats = self.PPG_phase(threshold)
            stats_actor_loss.append(Loss_actor_stats)
            stats_critic_loss.append(Loss_critic_stats)
            stats_loss_joint.append(Loss_joint_stats)
            stats_loss_aux.append(Loss_aux_stats)
            stats_reward.append(rew_stats)
            average, best = [],[]
            for rep in range(self.env.repetitions):
                rs = rew_stats[0][rep]
                max_reward = max(rs)
                avg_reward = sum(rs)/len(rs)
                average.append(avg_reward)
                best.append(max_reward)

            #stats_avg_reward.append(average)
            #stats_reward.append(best)

            lact = sum(Loss_actor_stats[0])/len(Loss_actor_stats[0])
            lcri = sum(Loss_critic_stats[0])/len(Loss_critic_stats[0])
            jjj = Loss_joint_stats[0]
            avgr = sum(average)/len(average)
            mr = sum(best)/len(best)
            print('Avg rew: {}, Max rew: {}, Loss actor: {:.2f}, Loss critic: {:.2f}, Loss Joint: {:.2f}'.format(avgr,mr,lact,lcri,jjj))
            self.Buffer.restart()
            if phase> self.phases/3:
                threshold = 0.1
            if phase> 2*self.phases/3:
                threshold = 0

            print('\n')

        return self.ACmodel.actor,self.ACmodel.critic,stats_reward,stats_actor_loss,stats_critic_loss,stats_loss_joint,stats_loss_aux

    def Test_Policy(self,actor,critic,threshold):
        s0 = self.env.initial_state()
        m0 = self.env.initial_mask(s0)
        stats = []
        for rep in range(self.env.repetitions,3*self.env.repetitions):
            B,cum_rew = self.policy_rollout(s0,m0,rep,actor,critic,threshold)
            stats.append(cum_rew)
        return stats


































