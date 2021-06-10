import torch
#torch.set_num_threads(4)
from torch import nn
from modules2 import CoreModel, GraphCNN, LstmModel, Basic_CNN, TSP_Model, CNN_st, CNN_st1, Encoder, MLP, TSP_Model_bis, Actor,Critic,Actor_GCN,Critic_GCN,Actor_GCN_base,Critic_GCN_base

class Model():
    def __init__(self, D_in, specs,edges,nnodes,mod_layers,LSTMflag = False, seed = None):
        #torch.set_num_threads(1)
        self.specs = specs
        #self.coremdl = Basic_CNN(D_in)
        self.coremdl = TSP_Model(edges=edges,nnodes=nnodes,batchsize=1,mod_layers=mod_layers)
        #self.coremdl = TSP_Model_bis(edges=edges,nnodes=nnodes,batchsize=1,mod_layers=mod_layers)
        #self.coremdl = GraphCNN(D_in,edges,nnodes,specs)
        #if edges is None:
            #if LSTMflag == True:
                #self.coremdl = LstmModel(D_in, specs)
            #else:
                #self.coremdl   = CoreModel(D_in, specs)
        #else:
            #if TSPflag == True:
                #self.coremdl = Basic_CNN()
            #else:
                #self.coremdl = GraphCNN(D_in, edges, nnodes, specs)
        if seed != None:
            torch.seed = seed


    def set_loss(self, losstype):
        if losstype == "mse":
            self.criterion = nn.MSELoss()
        elif losstype == "l1":
            self.criterion = nn.L1Loss()
        elif losstype == "smoothl1":
            self.criterion = nn.SmoothL1Loss()
        elif losstype == 'KLD':
            self.criterion = nn.KLDivLoss(reduction='batchmean')
        else:
            raise Exception("Invalid loss type")


    def set_optimizer(self, name, options):
        if name == "sgd":
            #print(self.coremdl.parameters())
            self.optimizer = torch.optim.SGD(self.coremdl.parameters(), lr=options["lr"], momentum=options["momentum"],nesterov=options["nesterov"])
        elif name == "adam":
            self.optimizer = torch.optim.Adam(self.coremdl.parameters(), lr=options["lr"])
        else:
            raise Exception("Invalid optimizer type")

    def set_scheduler(self, name, options):
        if name is None:
            self.scheduler = None
        elif name == "multiplicative":
            factor = options.get("factor") if options.get("factor") is not None else .99
            lmbda = lambda epoch : factor**epoch
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lmbda)

    def schedulerstep(self):
        if self.scheduler is not None:
            self.scheduler.step()
    #
    # def single_update(self, x, y):
    #     y_pred = self.coremdl(x)
    #     y = y.type(torch.FloatTensor)
    #     self.optimizer.zero_grad()
    #     self.criterion(y_pred, y).backward()
    #     self.optimizer.step()

    # def long_update(self, x, y, nsteps):
    #     for _ in range(nsteps):
    #         self.single_update(x, y)

    def single_update(self, x, y, bsize = None):
        y_pred = self.coremdl(x)
        y=y.type(torch.FloatTensor)
        self.optimizer.zero_grad()
        self.criterion(y_pred, y).backward()
        self.optimizer.step()

    def long_update(self, x, y, nsteps, bsize = None):
        for _ in range(nsteps):
            self.single_update(x, y, bsize)


class PPG_Model():

    def __init__(self,specsActor,specsCritic,device,GCNflag):
        self.device = device
        if GCNflag:
            self.actor = Actor_GCN(emb_size=specsActor['emb_size'], num_feat=1, device=self.device).to(self.device)
            self.critic = Critic_GCN(emb_size=specsCritic['emb_size'], num_feat=1, device=self.device).to(self.device)
        else:
            self.actor = Actor(conv_layers=specsActor['conv_layers'],
                fc_layers=specsActor['fc_layers'], maskdim=specsActor['maskdim'],device=self.device).to(self.device)
            self.critic = Critic(conv_layers=specsCritic['conv_layers'],
                fc_layers=specsCritic['fc_layers'],device=self.device).to(self.device)
        self.eps = specsActor['eps']
        self.beta = specsActor['beta']
        self.beta_c = specsActor['beta_c']
        self.lr_actor = specsActor['lr']
        self.lr_critic = specsCritic['lr']



    def Loss_actor(self,old_probs,probs,act_ind,advantages):
        batch_size = len(probs)
        loss = torch.empty(size=(batch_size,),device=self.device)
        for j in range(batch_size):
            probs_j = probs[j]
            old_probs_j = old_probs[j]
            ind_j = act_ind[j] # indice azione compiuta a j nella traiettoria
            ratio_j = probs_j[ind_j]/old_probs_j[ind_j]
            surr1 = ratio_j*advantages[j]
            surr2 = ratio_j.clamp(0.8,1.2)*advantages[j]
            dist = torch.distributions.Categorical(probs_j)
            entropy = dist.entropy()
            loss_j = torch.min(surr1,surr2)+self.beta*entropy
            loss[j] = loss_j

        return -torch.mean(loss)

    def Loss_critic(self,v_pred,v_target):
        l_value = nn.MSELoss()
        return l_value(v_pred.to(self.device),v_target.to(self.device))

    def Loss_joint(self,v_pred,v_target,old_probs,probs,bs):
        mse = nn.MSELoss()
        kl = nn.KLDivLoss()
        KL = torch.empty(size=(bs,),device=self.device)
        values_pred_tensor = torch.empty(size=(bs,),device=self.device)
        for j in range(bs):
            #print(old_probs[0])
            old_probs_j = old_probs[j]
            probs_j = probs[j]
            KL[j] = kl(old_probs_j,probs_j)
            values_pred_tensor[j] = v_pred[j]

        L_aux = self.Loss_critic(values_pred_tensor,torch.as_tensor(v_target))
        #print('KL_media = ',torch.mean(KL))

        return L_aux, L_aux + self.beta_c*torch.mean(KL)

    def set_loss(self):
        self.loss_critic = self.Loss_critic
        self.loss_actor = self.Loss_actor

    def set_optim(self):
        self.opt_actor = torch.optim.Adam(self.actor.parameters(),lr=self.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),lr=self.lr_critic)

    def update_actor(self,old_probs,probs,act_ind,advantages):
        self.opt_actor.zero_grad()
        loss = self.Loss_actor(old_probs,probs,act_ind,advantages)
        loss.backward(retain_graph=True)
        self.opt_actor.step()
        return loss

    def update_critic(self,v_pred,v_target):
        self.opt_critic.zero_grad()
        loss = self.loss_critic(v_pred,v_target)
        loss.backward()
        self.opt_critic.step()
        return loss


class PPO_Model():

    def __init__(self, specsActor, specsCritic, device, GCNflag):
        self.device = device
        if GCNflag:
            self.actor = Actor_GCN(emb_size=specsActor['emb_size'], num_feat=1, device=self.device)
            self.critic = Critic_GCN(emb_size=specsCritic['emb_size'], num_feat=1, device=self.device)
        else:
            self.actor = Actor(conv_layers=specsActor['conv_layers'],
                fc_layers=specsActor['fc_layers'], maskdim=specsActor['maskdim'],device=self.device)
            self.critic = Critic(conv_layers=specsCritic['conv_layers'],
                fc_layers=specsCritic['fc_layers'],device=self.device)

        self.eps = specsActor['eps']
        self.lr_actor = specsActor['lr']
        self.lr_critic = specsCritic['lr']
        self.beta = 1

    def Loss_actor(self,old_probs,probs,act_ind,advantages):
        batch_size = len(probs)
        loss = torch.empty(size=(batch_size,),device=self.device)
        for j in range(batch_size):
            probs_j = probs[j]
            old_probs_j = old_probs[j]
            ind_j = act_ind[j] # indice azione compiuta a j nella traiettoria
            ratio_j = probs_j[ind_j]/old_probs_j[ind_j]
            surr1 = ratio_j*advantages[j]
            surr2 = ratio_j.clamp(0.8,1.2)*advantages[j]
            dist = torch.distributions.Categorical(probs_j)
            entropy = dist.entropy()
            loss_j = torch.min(surr1,surr2)+self.beta*entropy
            loss[j] = loss_j

        return -torch.mean(loss)

    def Loss_critic(self,v_pred,v_target):
        l_value = nn.MSELoss()
        return l_value(v_pred,v_target)

    def set_loss(self):
        self.loss_critic = self.Loss_critic
        self.loss_actor = self.Loss_actor

    def set_optim(self):
        self.opt_actor = torch.optim.Adam(self.actor.parameters(),lr=self.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),lr=self.lr_critic)

    def update_actor(self,old_probs,probs,act_ind,advantages):
        self.opt_actor.zero_grad()
        loss = self.Loss_actor(old_probs,probs,act_ind,advantages)
        loss.backward(retain_graph=True)
        self.opt_actor.step()
        return loss

    def update_critic(self,v_pred,v_target):
        self.opt_critic.zero_grad()
        loss = self.loss_critic(v_pred,v_target)
        loss.backward()
        self.opt_critic.step()
        return loss







