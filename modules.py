import torch
torch.set_num_threads(4)
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
import math

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
        print(self.layers)

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
        return y_pred

class LstmModel(BasicModel): #o NNmodule
    def __init__(self, input_dim, specs):
        super(LstmModel, self).__init__()

        hidden_dim = specs[0]
        layer_dim = specs[1]
        output_dim = specs[2]
        layer2_dim = specs[3]
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, layer2_dim, batch_first=True)
        #self.out = nn.Linear(hidden_dim, output_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim),
            )

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, xx, bsize = 0):
        if bsize == 0:
             bsize = 1
             xx=[xx]
        out_m = torch.zeros((bsize))
        for j in range(bsize):
            x=xx[j]
            lsout = torch.zeros((1, len(x), self.hidden_dim))
            i = 0
            for xi in x:
                inp = torch.as_tensor([xi]).type(torch.FloatTensor)
                h0 = torch.zeros(self.layer_dim, inp.size(0), self.hidden_dim).requires_grad_()
                c0 = torch.zeros(self.layer_dim, inp.size(0), self.hidden_dim).requires_grad_()
                out, (hn, cn) = self.lstm(inp, (h0.detach(), c0.detach())) #L'out size sarà (batch, numero operazioni, hidden_dim), prendo ultima operazione
                #print("outsize", out[:, -1, :].reshape(-1).size())
                lsout[0][i] = out[:, -1, :].reshape(-1)
                i += 1
            h0 = torch.zeros(self.layer_dim, 1, self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.layer_dim, 1, self.hidden_dim).requires_grad_()  # todo fix the sizes in order to take into account mini-batches
            #lsout=lsout.reshape(-1)

            out, (hn, cn) = self.lstm2(lsout, (h0.detach(), c0.detach())) #lsout avrà dim (1, seq_lenght=numero job, hidden_dim)
            #out avrà (1, numero jobs, n features)
            # inp = torch.cat(lsout, -1)
            inp = out[:, -1, :].reshape(-1) #hidden dimension
            out = self.network(inp)
            out_m[j] = out #.detach()
        if bsize == 32:
            print('out_m', out_m)

        return out_m

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