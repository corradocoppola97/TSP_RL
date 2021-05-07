import torch
import torch.nn as nn
import copy

class Network(nn.Module):

    def __init__(self):
        super(Network,self).__init__()
        self.rete = nn.Sequential(nn.Linear(10,5),nn.ReLU(),nn.Linear(5,5),nn.ReLU(),nn.Linear(5,1))

    def forward(self,x):
        #x = x.float()
        x = self.rete(x)
        return x

miao = Network()
opt = torch.optim.Adam(miao.parameters(),lr=5e-2)
loss = nn.MSELoss()
x_p = torch.Tensor([-1.8803,0.2853,  0.8854, -0.2657, -0.9369,  0.3203,  0.0698,  0.3867,0.6141,  0.8825])
y_pred = torch.Tensor([5.3])
out = miao(x_p)
savemodel = copy.deepcopy(miao)
loss_value = loss(out,y_pred)
opt.zero_grad()
loss_value.backward()
opt.step()
out1 = miao(x_p)
out2 = savemodel(x_p)
