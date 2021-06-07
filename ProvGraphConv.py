import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import copy
from warnings import filterwarnings
filterwarnings('ignore')

mat_inc = (torch.ones(size=(5,5))-torch.eye(5)).float()
mat_inc_pes = torch.Tensor([[0,5,8,3,11],[5,0,6,9,13],[8,6,0,1,7],[3,9,0,1,9],[11,13,7,9,0]])

class GCN_prova(nn.Module):
    def __init__(self,emb_size,num_features):
        super(GCN_prova,self).__init__()
        self.conv1 = GCNConv(4,emb_size)
        self.conv2 = GCNConv(emb_size,emb_size)
        self.conv3 = GCNConv(emb_size,1)
        self.lin = nn.Linear(emb_size,1)

    def forward(self,x,A,p):
        print('x', x)
        print('x_shape', x.shape)
        x = self.conv1(x,A,p)
        print('x',x)
        print('x_shape',x.shape)
        x = self.conv2(x,A,p)
        print('x', x)
        print('x_shape', x.shape)
        x = self.conv3(x, A, p)
        print('x', x)
        print('x_shape', x.shape)
        x = x.flatten().softmax(0)
        print('prob',x)
        x = F.relu(x)
        x = self.lin(x)
        return x

rete = GCN_prova(emb_size=16,num_features=4)
l = []
for i in range(mat_inc_pes.shape[0]):
    vicini = mat_inc_pes[i]
    vicini_b = copy.deepcopy(vicini)
    vicini_b[i] = 1e6
    pot = [min(vicini_b).item(),max(vicini).item(),torch.mean(vicini).item(),torch.std(vicini).item()]
    l.append(pot)

x = torch.Tensor(l).float()
edge_ind = torch.empty(size=(20,2),dtype=torch.long)

edge = [(0,1),(0,2),(0,3),(0,4),(1,0),(1,2),(1,3),(1,4),(2,0),(2,1),(2,3),(2,4),(3,0),(3,1),(3,2),(3,4),(4,0),(4,1),(4,2),(4,3)]
for k in range(len(edge)):
    i,j = edge[k]
    edge_ind[k,0] = i
    edge_ind[k,1] = j

out = rete(x,edge_ind.t(),mat_inc_pes[mat_inc_pes>0])