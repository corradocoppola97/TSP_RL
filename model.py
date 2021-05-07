import torch
#torch.set_num_threads(4)
from torch import nn
from modules2 import CoreModel, GraphCNN, LstmModel, Basic_CNN, TSP_Model, CNN_st, CNN_st1, Encoder, MLP, TSP_Model_bis, Actor,Critic

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

    def __init__(self,specsActor,specsCritic):
        self.actor = Actor(conv_layers=specsActor['conv_layers'],
                           fc_layers=specsActor['fc_layers'],maskdim=specsActor['maskdim'])
        self.critic = Critic(conv_layers=specsCritic['conv_layers'],
                              fc_layers=specsCritic['fc_layers'])

        self.eps = specsActor['eps']
        self.beta = specsActor['beta']
        self.beta_c = specsActor['beta_c']
        self.lr_actor = specsActor['lr']
        self.lr_critic = specsCritic['lr']



    def Loss_actor(self,p_old,p,adv):
        eps = self.eps
        beta = self.beta
        ratio = p/p_old
        f1 = ratio*adv
        f2 = (ratio.clamp(1-eps,1+eps))*adv
        ob = torch.min(f1,f2)
        Lclip = torch.mean(ob)
        entropy = torch.distributions.Categorical(p).entropy() #Todo controllare entropia
        loss_actor = Lclip + beta*entropy
        return -loss_actor

    def Loss_value(self,st,Vtarg):
        l_value = nn.MSELoss()
        V_pred = self.critic(st)
        return -l_value(Vtarg,V_pred)

    def Loss_joint(self,p_old,p,Vtarg,Vpred):
        mse = nn.MSELoss()
        kl = nn.KLDivLoss()
        loss_aux = mse(Vtarg,Vpred)
        return -(loss_aux + self.beta_c*kl(p,p_old))

    def set_loss(self):
        self.loss_critic = self.Loss_value
        self.loss_actor = self.Loss_actor

    def set_optim(self):
        self.opt_actor = torch.optim.Adam(self.actor.parameters(),lr=self.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),lr=self.lr_critic)

    def update_actor(self,p_old,p,adv):
        self.opt_actor.zero_grad()
        lossactor = self.loss_actor(p_old,p,adv)
        lossactor.backward()
        self.opt_actor.step()
        #print(lossactor)
        return lossactor

    def update_critic(self,st,V_target):
        self.opt_critic.zero_grad()
        self.loss_critic(st,V_target).backward()
        self.opt_critic.step()









