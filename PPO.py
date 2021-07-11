from model import PPO_Model
from modules2 import Actor,Critic
from data_manager import PPG_data_manager,PPG_data_manager_NO_GCN
from Environment_GH import environment, EnvSpecs, EnvType, tsp
import copy
import random
import torch
from support import build_index,build_features,build_attr
import time

class ppo():

    def __init__(self,nit,
                 epochs,
                 exper,
                 batchsize,
                 beta,
                 specsActor,
                 specsCritic,
                 device,GCNflag):

        self.device = device
        self.GCNflag = GCNflag
        self.nit = nit
        self.epochs = epochs
        self.exper = exper
        self.batchsize = batchsize
        self.beta = beta
        self.specsActor = specsActor
        self.specsCritic = specsCritic
        self.ACmodel = PPO_Model(specsActor,specsCritic,GCNflag=self.GCNflag,device=self.device)
        self.ACmodel.set_loss()
        self.ACmodel.set_optim()

    def buildenvironment(self,envspecs):
        self.env = tsp(envspecs)
        self.rollout_lenght = self.env.nnodes
        if self.GCNflag:
            self.Buffer = PPG_data_manager(stacklenght=50000)
        else:
            self.Buffer = PPG_data_manager_NO_GCN(stacklenght=50000)

    def perform_rollouts(self,policy,s0,m0,threshold,greedy=False):
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
            if self.GCNflag:
                edge_ind = build_index(ns)
                features = build_features(mat_inc_st,f_insts[mt[0]][0][1],self.env.incidence_matrix,f_insts[mt[0]][0][0])
                edge_indexes_list.append(edge_ind)
                features_list.append(features)
                action_probs = policy([features],[edge_ind],[edge_attr])[0]
                edge_attr = build_attr(mat_inc_st)
                edge_attr_list.append(edge_attr)
            else:
                action_probs = policy([torch.as_tensor(mat_inc_st)], [mt])[0]
            pol_probs.append(action_probs)
            dist_act_prob = torch.distributions.Categorical(action_probs)
            eps = random.random()
            if eps<=threshold:
                act_ind = random.randint(0,len(action_probs)-1)
                #print(act_ind)
            else:
                act_ind = dist_act_prob.sample().item()
            if greedy:
                act_ind = torch.argmax(action_probs).item()
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
        if self.GCNflag:
            return rew,pol_probs,states,mask_list,cum_rew,action_list,actions_ind_list,edge_indexes_list,features_list,edge_attr_list
        else:
            return rew, pol_probs, states, mask_list, cum_rew, action_list, actions_ind_list

    def policy_rollout(self,s0,m0,rep,cp,crr,threshold,greedy=False):
        self.env.set_instance(rep)
        if self.GCNflag:
            B = {'state': [], 'actions_probs': [], 'advantages':[],'values':[],'masks':[],'actions':[],'actions_ind':[],
                'edge_indexes_list':[],'features_list':[], 'edge_attr_list':[]}
            rew, pol_probs, states, masks, cum_rew, action_list, actions_ind_list, eil, fl, eal = self.perform_rollouts(cp, s0, m0, threshold)
        else:
            B = {'state': [], 'actions_probs': [], 'advantages': [], 'values': [], 'masks': [], 'actions': [],'actions_ind': []}
            rew, pol_probs, states, masks, cum_rew, action_list, actions_ind_list = self.perform_rollouts(cp, s0, m0, threshold,greedy)

        cr = cum_rew.copy()
        cr.reverse()
        V_target_list = cr
        if self.GCNflag:
            V_pred_list = [crr([fl[k]], [eil[k]], [eal[k]])[0] for k in range(len(fl))]
        else:
            V_pred_list = [crr([torch.as_tensor(elem)])[0] for elem in states]
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
            if self.GCNflag:
                B['edge_indexes_list'].append(eil[k])
                B['features_list'].append(fl[k])
                B['edge_attr_list'].append(eal[k])
        return B,cum_rew

    def ppo_algo(self,envspecs,threshold,file_name):
        path = '\\Users\corra\OneDrive\Desktop\Tesi\ModelliSalvati'
        self.buildenvironment(envspecs)
        s0 = self.env.initial_state()
        m0 = self.env.initial_mask(s0)
        stats_reward = []
        #stats_avg_reward = []
        stats_loss_actor = []
        stats_loss_critic = []
        for i in range(self.nit):
            if i>= 1700:
                threshold = 0.1
            if i>= 3800:
                threshold = 0
            print('Inizio iterazione {} di {}'.format(i+1,self.nit))
            current_policy = copy.deepcopy(self.ACmodel.actor)
            current_critic = copy.deepcopy(self.ACmodel.critic)
            iteration_rew_stats = []
            best_stats = []
            avg_stats = []
            for rep in range(self.env.repetitions):
                rep_stats = []
                for k in range(self.exper):
                    B, cum_rew = self.policy_rollout(s0,m0,rep,current_policy,current_critic,threshold)
                    self.Buffer.add(B)
                    rep_stats.append(cum_rew[-1])
                avg_rep = sum(rep_stats)/len(rep_stats)
                best_rep = max(rep_stats)
                best_stats.append(best_rep)
                avg_stats.append(avg_rep)
                iteration_rew_stats.append(rep_stats)
            print('Avg Rew: {},   Best Rew: {}'.format(sum(avg_stats)/len(avg_stats),sum(best_stats)/len(best_stats)))
            stats_reward.append(iteration_rew_stats)

            Loss_Actor_Stats = []
            Loss_Critic_Stats = []
            #self.ACmodel.set_loss()
            #self.ACmodel.set_optim()
            for epoch in range(self.epochs):
                print('Epoch n. {} di {}'.format(epoch+1,self.epochs))
                la,lc = [],[]
                num_iter_epoch = int(self.Buffer.get_lenght()/self.batchsize)
                l_ind = [_ for _ in range(self.Buffer.get_lenght())]
                for ii in range(num_iter_epoch):
                    list_index = l_ind[ii*self.batchsize:min((ii+1)*self.batchsize,len(l_ind)-1)]
                    if self.GCNflag:
                        old_action_probs, values_t, action_inds, adv, states_list, masks_list, edg_inds, feat_list, edge_ats = self.Buffer.get_batch(self.batchsize)
                    else:
                        old_action_probs, values_t, action_inds, adv, states_list, masks_list = self.Buffer.get_batch(self.batchsize)
                    values_target = torch.as_tensor(values_t)
                    adv = torch.as_tensor(adv)
                    values_pred = torch.empty(size=(self.batchsize,))
                    if self.GCNflag:
                        action_probs = self.ACmodel.actor(feat_list, edg_inds, edge_ats)
                        values = self.ACmodel.critic(feat_list, edg_inds, edge_ats)
                    else:
                        action_probs = self.ACmodel.actor(states_list, masks_list)
                        values = self.ACmodel.critic(states_list)
                    loss_actor = self.ACmodel.update_actor(old_action_probs, action_probs, action_inds, adv)
                    for k in range(len(values)):
                        values_pred[k] = values[k]

                    loss_critic = self.ACmodel.update_critic(values_pred, values_target)
                    la.append(loss_actor.item())
                    lc.append(loss_critic.item())

                la_r, lc_r = sum(la)/len(la), sum(lc)/len(lc)
                Loss_Actor_Stats.append(la_r)
                Loss_Critic_Stats.append(lc_r)
                print('Loss Actor: {:.2f}      Loss Critic: {:.2f}'.format(la_r,lc_r))


            stats_loss_actor.append(Loss_Actor_Stats)
            stats_loss_critic.append(Loss_Critic_Stats)
            modelpath_policy = path + file_name[0]
            modelpath_value = path + file_name[1]

            torch.save({
                'episode': i+1,
                'model_state_dict': self.ACmodel.actor.state_dict(),
                'optimizer_state_dict': self.ACmodel.opt_actor.state_dict(),
                'loss': Loss_Actor_Stats[-1],
            }, modelpath_policy)

            torch.save({
                'episode': i + 1,
                'model_state_dict': self.ACmodel.critic.state_dict(),
                'optimizer_state_dict': self.ACmodel.opt_critic.state_dict(),
                'loss': Loss_Critic_Stats[-1],
            }, modelpath_value)
            self.Buffer.restart()

        return stats_loss_actor,stats_loss_critic,stats_reward, self.ACmodel.actor, self.ACmodel.critic

    def Test_Policy(self, actor, critic, threshold, tenv, num_rollouts=10, greedy=False):
        self.buildenvironment(tenv)
        s0 = self.env.initial_state()
        m0 = self.env.initial_mask(s0)
        stats = []
        times = []
        for rep in range(self.env.repetitions):
            rep_stats = []
            if greedy==False:
                for h in range(num_rollouts):
                    t1 = time.time()
                    B, cum_rew = self.policy_rollout(s0, m0, rep, actor, critic, threshold)
                    t2 = time.time()
                    rep_stats.append(cum_rew[-1])
                    if h==0:
                        times.append(t2-t1)
                stats.append(rep_stats)
            else:
                B, cum_rew = self.policy_rollout(s0, m0, rep, actor, critic, threshold, greedy)
                stats.append(cum_rew[-1])

        return stats,times

    def load_models(self, path, file_name, nep_model=None, test_flag=False):
        modelpath_policy = path + file_name[0]
        modelpath_value = path + file_name[1]

        checkpoint = torch.load(modelpath_policy)
        self.ACmodel.actor.load_state_dict(checkpoint['model_state_dict'])
        self.ACmodel.opt_actor.load_state_dict(checkpoint['optimizer_state_dict'])

        checkpoint = torch.load(modelpath_value)
        self.ACmodel.critic.load_state_dict(checkpoint['model_state_dict'])
        self.ACmodel.opt_critic.load_state_dict(checkpoint['optimizer_state_dict'])

        if test_flag:
            self.ACmodel.actor.eval()
            self.ACmodel.critic.eval()
        else:
            self.ACmodel.actor.train()
            self.ACmodel.critic.train()

        return self.ACmodel.actor,self.ACmodel.critic












