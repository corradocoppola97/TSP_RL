import torch
#torch.set_num_threads(4)
from torch import nn
from modules2 import CoreModel, GraphCNN, LstmModel

class Model():
    def __init__(self, D_in, specs,  LSTMflag = False, seed = None, edges = None, nnodes = None):
        #torch.set_num_threads(1)
        self.specs = specs
        if edges is None:
            if LSTMflag == True:
                self.coremdl = LstmModel(D_in, specs)
            else:
                self.coremdl   = CoreModel(D_in, specs)
        else:
            self.coremdl = GraphCNN(D_in, edges, nnodes, specs)
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

    def single_update(self, x, y, bsize = 0):
        y_pred = self.coremdl(x, bsize)
        y=y.type(torch.FloatTensor)
        self.optimizer.zero_grad()
        self.criterion(y_pred, y).backward()
        self.optimizer.step()

    def long_update(self, x, y, nsteps, bsize = 0):
        for _ in range(nsteps):
            self.single_update(x, y, bsize)