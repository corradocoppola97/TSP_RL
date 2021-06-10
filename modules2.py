import torch
#torch.set_num_threads(4)
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import copy
#special layer
class LogSumExp(nn.Module):
    def __init__(self,in_features, out_features,bias = False):
        super(LogSumExp, self).__init__()
        self.n = in_features
        self.N = out_features
        self.weight = Parameter(torch.Tensor(self.N, self.n))
        # if bias:
        #     self.bias = Parameter(torch.Tensor(out_features))
        # else:
        #     self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        size = x.size()
        if len(size) == 1:
            B = 1
            pW = torch.stack([torch.stack([ self.weight[i]*torch.as_tensor([x[ll] for ll in range(self.n)]) for b in range(B)])for i in range(self.N)])
            ret = torch.reshape(torch.logsumexp(pW, dim=2), (B,self.N))
        else:
            B = len(x)
            pW = torch.stack([torch.stack([self.weight[i] * torch.as_tensor([x[b][ll] for ll in range(self.n)]) for b in range(B)]) for i in range(self.N)])
            ret = torch.logsumexp(pW, dim=2).transpose(0,1)
        return ret
#superclass for the model that can be used inside the algorithm
class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.layers = []
        self.depth = 0
        self.specialind = -1
    def forward(self):
        pass

    def _codify_layers(self,  i, specs, mB = None):
        spec0 = specs[i]
        spec = specs[i + 1]
        biasflag =  mB is None
        if spec[0] == "linear":
            self.layers.append(nn.Linear(spec0[1], spec[1], bias = biasflag))
        elif spec[0] == "relu":
            db = nn.Linear(spec0[1], spec[1], bias = biasflag)
            db2 = nn.ReLU()
            self.layers.append(db)
            self.layers.append(db2)
            self.depth = self.depth + 1
        elif spec[0] == "leakyrelu":
            db = nn.Linear(spec0[1], spec[1], bias = biasflag)
            db2 = nn.LeakyReLU()
            self.layers.append(db)
            self.layers.append(db2)
            self.depth = self.depth + 1
        elif spec[0] == "sigmoid":
            db = nn.Linear(spec0[1], spec[1], bias = biasflag)
            db2 = nn.Sigmoid()
            self.layers.append(db)
            self.layers.append(db2)
            self.depth = self.depth + 1
        elif spec[0] == "logsigmoid":
            db = nn.Linear(spec0[1], spec[1], bias = biasflag)
            db2 = nn.LogSigmoid()
            self.layers.append(db)
            self.layers.append(db2)
            self.depth = self.depth + 1
        elif spec[0] == "softmax":
            db = nn.Linear(spec0[1], spec[1], bias = biasflag)
            db2 = nn.Softmax(dim=0)
            self.layers.append(db)
            self.layers.append(db2)
            self.depth = self.depth + 1
        elif spec[0] == "logsumexp":
            db = LogSumExp(spec0[1], spec[1])
            self.layers.append(db)
        elif spec[0] == "approximator":
            val = 2 * round(spec[1] / 3)
            valres = spec[1] - val
            db = nn.Linear(spec0[1], valres)
            self.layers.append(db)
            self.specialLayer = nn.Linear(spec0[1], val)
            self.specialLayer2 = nn.Sigmoid()
            self.specialLayer3 = nn.Dropout()
            self.specialind = len(self.layers) - 1
        elif spec[0] == "maxpool":
            self.layers.append(nn.MaxPool1d(spec0[1], spec[1]))
        elif spec[0] == "softmin":
            self.layers.append(nn.Softmin(spec0[1], spec[1]))
        else:
            raise Exception("Not a valid input class for the layer.")
class CoreModel(BasicModel):
    def __init__(self, D_in, specs):
        super(CoreModel, self).__init__()
        self.specialind = -1
        self.depth = len(specs)
        specs = [("", D_in)] + specs
        self.layers = []
        for i in range(self.depth):
            self._codify_layers(i, specs, mB = True)
        self.layers = nn.ModuleList(self.layers)
        #print(self.layers)

    def forward(self, x):
        y_pred = self.layers[0](x)


        for i in range(1, self.depth):
            if i != self.specialind:
                # if i <= 2:
                #     with torch.no_grad():
                #         y_pred = self.layers[i](y_pred)
                y_pred = self.layers[i](y_pred)
            else:
                y_predA = self.layers[i](y_pred)
                y_predB = self.specialLayer(y_pred)
                y_predB = self.specialLayer2(y_predB)
                y_predB = self.specialLayer3(y_predB)
                y_pred = torch.cat((y_predA,y_predB), dim = 1)
        #print(y_pred)
        return y_pred
class LstmModel(BasicModel): #o NNmodule
    def __init__(self, input_dim, specs):
        super(LstmModel, self).__init__()

        hidden_dim = specs[0]
        layer_dim = specs[1]
        output_dim = specs[2]
        layer2_dim = specs[3]
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim*2, layer2_dim, batch_first=True)#, bidirectional = True)
        #self.out = nn.Linear(hidden_dim, output_dim)
        #self.network = nn.Linear(hidden_dim*2, output_dim)

        self.network = nn.Sequential(
            nn.Linear(hidden_dim*2,hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, xx, bsize = 0):
        if bsize == 0:
             xx=[xx]
        megapad = [torch.tensor(xxxi).type(torch.FloatTensor) for xxi in xx for xxxi in xxi]
        lens0 = [len(m) for m in xx]
        megapad = pad_sequence(megapad, batch_first=True)
        megapad = [[megapad[i].numpy() for i in range(sum(lens0[h] for h in range(k)), sum(lens0[h] for h in range(k + 1)))] for k in range(len(lens0))]
        megapad = [torch.tensor(minipad).type(torch.FloatTensor) for minipad in megapad]
        megapad = pad_sequence(megapad, batch_first=True)
        minibatch = torch.zeros((megapad.size(0),  megapad.size(1), self.hidden_dim ))
        for i in range(megapad.size(1)):
            inp = megapad[:,i]
            out = self.lstm(inp)[0]  # , (h0.detach(), c0.detach()))
            minibatch[:,i] = out[:,-1]#torch.sum(out, axis=1)  # out[:,-1,:]
        out = self.lstm2(minibatch)[0]  # , (h0.detach(), c0.detach()))
        out = self.network(torch.sum(out, axis=1))  # out[:,-1,:])
        return out




        #
        # if bsize == 0:
        #      bsize = 1
        #      xx=[xx]
        # out_m = torch.zeros((bsize))
        #
        # for j in range(bsize):
        #     x=xx[j]
        #     lsout = torch.zeros((1, len(x), self.hidden_dim))
        #     x = [torch.as_tensor(xi).type(torch.FloatTensor) for xi in x]
        #     x = pad_sequence(x, batch_first = True, padding_value = 0)
        #     h0 = torch.zeros(self.layer_dim, len(x), self.hidden_dim).requires_grad_()
        #     c0 = torch.zeros(self.layer_dim, len(x), self.hidden_dim).requires_grad_()
        #     out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        #     lsout[0] = out[:, -1, :]
        #     h0 = torch.zeros(self.layer_dim, 1, self.hidden_dim*2).requires_grad_()
        #     c0 = torch.zeros(self.layer_dim, 1, self.hidden_dim*2).requires_grad_()  # todo fix the sizes in order to take into account mini-batches
        #     out2, (hn, cn) = self.lstm2(lsout, (h0.detach(), c0.detach()))
        #     inp = torch.sum(out2, axis=1).reshape(-1)
        #
        #     out = self.network(inp)
        #     out_m[j] = out
        # return out_m
class GraphCNN(BasicModel):
    def __init__(self, D_in,edges, nnodes,specs):
        # the last element should contain D_out
        super(GraphCNN, self).__init__()
        self.adjacency_matrix = torch.zeros((nnodes, nnodes))
        self.special_incidence = torch.zeros(nnodes, 4*(len(edges)))

        h = 0
        for (i,j) in edges:
            self.adjacency_matrix[i][j] = 1
            for ll in range(4):
                self.special_incidence[i][h+ll*len(edges)] = -1
                self.special_incidence[j][h+ll*len(edges)] = 1
            h = h+1
        self.special_incidence_transpose = torch.transpose(self.special_incidence, -1, 0)
        self.specialind = -1
        self.depth = len(specs)
        specs = [("", D_in)] + specs
        self.layers = []
        self._codify_layers(i=0, specs=specs, mB=True)
        depth = len(specs)-1
        for i in range(1,depth):
            self._codify_layers( i, specs=specs)
        self.layers = nn.ModuleList(self.layers)

        self.mA = [None for ind in range(self.depth)]
        self.mA[2] = True
        #self.mA[4] = True
        self.mA[6] = True
        #self.mA[8] = True
        #self.mA[10] = True
        #self.mA[self.depth - 3] = True
        #self.mA[self.depth - 1] = True

        #print(self.layers)

    def forward(self, x):
        x = x.float()
        # self.layers[0].weight = nn.Parameter(self.layers[0].weight*self.special_incidence)
        y_pred = self.layers[0](x)
        for i in range(1, self.depth - 1):
            if i != self.specialind:
                if self.mA[i] is not None:
                    self.layers[i].weight = nn.Parameter(self.layers[i].weight * self.adjacency_matrix)
                y_pred = self.layers[i](y_pred)
            else:
                y_predA = self.layers[i](y_pred)
                y_predB = self.specialLayer(y_pred)
                y_predB = self.specialLayer2(y_predB)
                y_predB = self.specialLayer3(y_predB)
                y_pred = torch.cat((y_predA,y_predB), dim = 1)

        if self.mA[self.depth-1] is not None:
            #print("LAST",self.layers[self.depth-1].weight)
            self.layers[self.depth-1].weight = nn.Parameter(self.layers[self.depth-1].weight * self.special_incidence_transpose)
        y_pred = self.layers[self.depth-1](y_pred)
        return y_pred
class Basic_CNN(nn.Module):

    def __init__(self,D_input):
        super(Basic_CNN,self).__init__()
        self.D_input = D_input
        self.fc1 = nn.Linear(self.D_input,self.D_input*2) #[1,16,16]
        #self.conv1 = nn.Conv2d(1,10,3,stride=1,padding=1) #[10,16,16] -> [10,8,8]
        #self.conv2 = nn.Conv2d(10,20,3,stride=1,padding=1) #[20,8,8] -> [20,4,4]
        self.fc2 = nn.Linear(self.D_input*2,self.D_input)
        self.fc3 = nn.Linear(self.D_input,1)
        #self.fc4 = nn.Linear(50,10)
        #self.fc5 = nn.Linear(10,1)
        self.RL1 = nn.ReLU()
        self.RL2 = nn.ReLU()


    def forward(self,x):
        x = x.float()
        x = self.RL1(self.fc1(x))
        #x = F.sigmoid(x)
        # = x.view(-1,1,16,16)
        #x = F.avg_pool2d(self.conv1(x),kernel_size=(2,2))
        #x = F.avg_pool2d(self.conv2(x),kernel_size=(2,2))
        #x = x.view(-1,20*4*4)
        x = self.RL2(self.fc2(x))
        x = self.fc3(x)
        #x = F.sigmoid(self.fc4(x))
        #x = self.fc5(x)
        return x

class TSP_Model(nn.Module):

    def __init__(self,edges,nnodes,batchsize,mod_layers):
        super(TSP_Model,self).__init__()
        self.layers_CNNst = mod_layers['CNNst']
        self.layers_CNNst1 = mod_layers['CNNst1']
        self.layers_MLP = mod_layers['MLP']
        self.foc = self.layers_CNNst[-1].out_channels
        self.maxout_CNNst = int((nnodes+1)**2)
        self.maxout_CNNst1 = int((nnodes)**2)
        self.fixed_dim = self.maxout_CNNst1 + self.maxout_CNNst + self.layers_MLP[-1].out_features
        self.layers_encoder = mod_layers['encoder']
        self.action_dim = len(edges)
        self.conv_st = CNN_st(layers=self.layers_CNNst)
        self.conv_st1 = self.conv_st
        self.ffn = MLP(layers=self.layers_MLP)
        self.encoder = Encoder(layers=self.layers_encoder)
        self.bs = batchsize


    def forward(self,inst_batch):
        outputs = None
        #print(len(inst_batch))
        #print(len(inst_batch))
        for j in range(self.bs):
            inst = inst_batch[j]
            #print(inst)
            if len(inst)==6:
                at,st,st1 = inst[5],inst[3],inst[4] #[0,0,0,0,1,0,0,0,0] <---> at.index(1) e ottengo indice arco
                st = torch.as_tensor(st).float()
                at = torch.as_tensor(at).float()
                st1 = torch.as_tensor(st1).float()
            elif len(inst)==3:
                st,at,st1 = inst[0],inst[1],inst[2]
            #print(st.shape, at.shape, st1.shape)
            out_st = self.conv_st(st)
            #print(out_st.shape)
            out_at = self.ffn(at)
            #print(out_at.shape)
            out_st1 = self.conv_st1(st1)
            out_st = out_st.view(self.foc,out_st.shape[2],out_st.shape[3])
            out_st1 = out_st1.view(self.foc, out_st1.shape[2], out_st1.shape[3])
            out_st = torch.sum(out_st,axis=0).flatten()
            out_st1 = torch.sum(out_st1,axis=0).flatten()
            #out_st = out_st.flatten()
            #out_st1 = out_st1.flatten()
            st_shape, st1_shape = out_st.shape[0], out_st1.shape[0]
            if st_shape<self.maxout_CNNst:
                out_st = torch.cat((out_st,torch.zeros(self.maxout_CNNst-st_shape)))
            if st1_shape<self.maxout_CNNst1:
                out_st1 = torch.cat((out_st1,torch.zeros(self.maxout_CNNst1-st1_shape)))
            input_encoder = torch.cat((out_st,out_at,out_st1))
            #Concatena per output di ciascuna rete rispetto a dimensione massima uscita
            out = self.encoder(input_encoder)
            if outputs is None:
                outputs = out
            else:
                outputs = torch.cat((outputs,out))


        #print(out)
        return out

class TSP_Model_bis(nn.Module):

    def __init__(self,edges,nnodes,batchsize,mod_layers):
        super(TSP_Model_bis,self).__init__()
        self.layers = mod_layers
        self.bs = batchsize
        self.maxout = int(nnodes + 1 + (nnodes+1)**2 + nnodes**2)
        self.maxSt = int((nnodes+1)**2)
        self.maxSt1 = int(nnodes**2)

    def forward(self,inst_batch):
        outputs = None
        for j in range(self.bs):
            inst = inst_batch[j]
            # print(inst)
            if len(inst)==6:
                at, st, st1 = inst[5], inst[3], inst[4]  # [0,0,0,0,1,0,0,0,0] <---> at.index(1) e ottengo indice arco
                st = torch.as_tensor(st).float()
                at = torch.as_tensor(at).float()
                st1 = torch.as_tensor(st1).float()
            elif len(inst)==3:
                st, at, st1 = inst[0], inst[1], inst[2]

            st = st.flatten()
            st1 = st1.flatten()
            st_shape, st1_shape = st.shape[0], st1.shape[0]
            if st_shape < self.maxSt:
                st = torch.cat((st,torch.zeros(self.maxSt-st_shape)))
            if st1_shape < self.maxSt1:
                st1 = torch.cat((st1,torch.zeros(self.maxSt1-st1_shape)))

            input_en = torch.cat((st,at,st1)).float()
            out = self.layers(input_en)
            if outputs is None:
                outputs = out
            else:
                outputs = torch.cat((outputs,out))
        return outputs

class CNN_st(nn.Module):

    def __init__(self,layers):
        super(CNN_st,self).__init__()
        self.layers = layers

    def forward(self,x):
        x = x.float()
        dim_in = x.shape[0]
        #print('xsh', x.shape)
        #print(x)
        x = x.view(-1,1,dim_in,dim_in)
        out = self.layers(x)
        return out

class CNN_st1(nn.Module):

    def __init__(self,layers):
        super(CNN_st1,self).__init__()
        self.layers = layers

    def forward(self,x):
        x = x.float()
        dim_in = x.shape[0]
        x = x.view(-1,1,dim_in,dim_in)
        out = self.layers(x)
        return out

class Encoder(nn.Module):

    def __init__(self,layers):
        super(Encoder,self).__init__()
        self.layers = layers
        #d = dim_in
        #neurons = [d]
        #while d>=10:
            #d = d//2
            #neurons.append(d)

        #self.layers = nn.Sequential()
        #for j in range(len(neurons)-1):
            #fc = nn.Linear(neurons[j],neurons[j+1])
            #self.layers.add_module('fc'+str(j+1),fc)
            #self.layers.add_module('F'+str(j+1),nn.Sigmoid())

        #self.layers.add_module('fcf', nn.Linear(neurons[len(neurons)-1],1))

    def forward(self,x):
        x = x.float()
        #print(self.layers)
        out = self.layers(x)
        return out

class MLP(nn.Module):

    def __init__(self,layers):
        super(MLP,self).__init__()
        self.layers = layers

    def forward(self,x):
        x = x.float()
        out = self.layers(x)
        return out

class Actor(nn.Module):

    def __init__(self,conv_layers,fc_layers,maskdim,device):
        super(Actor,self).__init__()
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.maxout = fc_layers[0].in_features
        self.maskdim = maskdim
        self.device = device


    def forward(self,x,mask_list):
        bs = len(x)
        out = []
        for j in range(bs):
            #print(bs)
            #print(x[j].shape)
            ns = x[j].shape[1]
            xj = torch.as_tensor(x[j],device=self.device).float()
            xj = xj.view(-1,1,ns,ns)
            xj = self.conv_layers(xj)
            xj = xj.flatten()
            xs_j = xj.shape[0]
            if xs_j<self.maxout:
                xj = torch.cat((xj,torch.zeros(self.maxout-xs_j,device=self.device)))
            xj = self.fc_layers(xj)
            xj = xj[mask_list[j]]
            xj = xj.softmax(0)
            out.append(xj)
        return out

class Critic(nn.Module):

    def __init__(self,conv_layers,fc_layers,device):
        super(Critic,self).__init__()
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.maxout = fc_layers[0].in_features
        self.device = device


    def forward(self,x):
        bs = len(x)
        out = []
        for j in range(bs):
            #print(x[j].shape)
            ns = x[j].shape[1]
            xj = torch.as_tensor(x[j],device=self.device).float()
            xj = xj.view(-1,1,ns,ns)
            xj = self.conv_layers(xj)
            xj = xj.flatten()
            xj_s = xj.shape[0]
            if xj_s<self.maxout:
                xj = torch.cat((xj,torch.zeros(self.maxout-xj_s,device=self.device)))
            xj = self.fc_layers(xj)
            out.append(xj)
        return out


class Actor_GCN(nn.Module):

    def __init__(self,emb_size,num_feat,device):
        super(Actor_GCN,self).__init__()
        self.num_feat = num_feat
        self.emb_size = emb_size
        self.conv1 = GCNConv(self.num_feat,emb_size,add_self_loops=False)
        self.conv2 = GCNConv(emb_size,emb_size,add_self_loops=False)
        self.conv22 = GCNConv(emb_size, emb_size,add_self_loops=False)
        self.conv3 = GCNConv(emb_size,1,add_self_loops=False)
        self.device = device

    def forward(self,batch_feat,batch_edges,batch_attr):
        bs = len(batch_feat)
        nsb = batch_feat[0].shape[0]
        out = []
        for j in range(bs):
            x, edge_ind, adj = batch_feat[j].float(), batch_edges[j], batch_attr[j]
            #mask = copy.deepcopy(x).flatten().to(self.device)
            x,edge_ind,adj = x.to(self.device),edge_ind.to(self.device),adj.to(self.device)
            x = self.conv1(x,edge_ind,adj)
            x = F.relu(x)
            x = self.conv2(x,edge_ind,adj)
            x = F.relu(x)
            x = self.conv22(x,edge_ind,adj)
            x = F.relu(x)
            x = self.conv3(x,edge_ind,adj)

            x = x.flatten()
            #mask = torch.as_tensor(mask,dtype=torch.bool,device=self.device)
            #x = x[mask].softmax(0)
            #if mask.shape[0]<=2:
                #x = torch.Tensor([1.0]).softmax(0)
            if x.shape[0]>2:
                x = x[1:x.shape[0]-1].softmax(0)
            else:
                x = torch.Tensor([1.0],device=self.device).softmax(0)

            out.append(x)

        return out


class Critic_GCN(nn.Module):

    def __init__(self,emb_size,num_feat,device):
        super(Critic_GCN,self).__init__()
        self.num_feat = num_feat
        self.emb_size = emb_size
        self.conv1 = GCNConv(self.num_feat,emb_size,add_self_loops=False)
        self.conv2 = GCNConv(emb_size,emb_size,add_self_loops=False)
        self.conv3 = GCNConv(emb_size,emb_size,add_self_loops=False)
        self.linear = nn.Linear(emb_size,1)
        self.device = device

    def forward(self,batch_feat,batch_edges,batch_attr):
        bs = len(batch_feat)
        out = []
        for j in range(bs):
            x, edge_ind, adj = batch_feat[j].float(), batch_edges[j], batch_attr[j]
            #print(x,edge_ind,adj)
            x, edge_ind, adj = x.to(self.device), edge_ind.to(self.device), adj.to(self.device)
            x = self.conv1(x,edge_ind,adj)
            x = F.relu(x)
            x = self.conv2(x,edge_ind,adj)
            x = F.relu(x)
            x = F.relu(self.conv3(x,edge_ind,adj))
            x = x.mean(0)
            #print('X_shape',x.shape)
            x = self.linear(x)
            #print('x_fin',x)
            out.append(x)

        return out


class Actor_GCN_base(nn.Module):

    def __init__(self,dim_in,device):
        super(Actor_GCN_base,self).__init__()
        self.dim_in = dim_in
        self.device = device
        self.network = nn.Sequential(nn.Linear(dim_in,10*dim_in),nn.ReLU(),nn.Linear(10*dim_in,dim_in)).to(self.device)

    def forward(self,batch_feat,m,b):
        bs = len(batch_feat)
        out = []
        for j in range(bs):
            x,cd = batch_feat[j]
            x = x.float().to(self.device)
            x = self.network(x)
            mask = [1.0 for _ in range(self.dim_in)]
            mask = torch.as_tensor(mask,dtype=torch.bool,device=self.device)
            mask[-1] = 0
            mask[cd] = 0
            x = x[mask]
            if x.shape[0]>=1:
                x = x.softmax(0)
            else:
                x = torch.as_tensor([1.0],dev).softmax(0)
            out.append(x)
        return out

class Critic_GCN_base(nn.Module):

    def __init__(self,dim_in,device):
        super(Critic_GCN_base,self).__init__()
        self.dim_in = dim_in
        self.device = device
        self.network = nn.Sequential(nn.Linear(dim_in,10*dim_in),nn.ReLU(),nn.Linear(10*dim_in,1)).to(self.device)

    def forward(self,batch_feat,m,b):
        bs = len(batch_feat)
        out = []
        for j in range(bs):
            x,cd = batch_feat[j]
            x = x.float().to(self.device)
            x = self.network(x)
            out.append(x)

        return out




