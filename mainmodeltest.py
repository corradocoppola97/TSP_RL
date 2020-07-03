
from model import Model, CoreModel
import torch
#from torch.nn import
import copy

import random
from torch import nn



nedges = 10
D_in = 10
D_out = 1
specs = [
    ("relu", nedges),("relu", nedges),("relu", nedges),
    #("relu", nnodes),
    #("logsigmoid", nedges),
    ("linear", D_out)
]
# print(specs)
criterion = "mse"
optimizer = "sgd"
optspecs = { "lr" : 1e-3, "momentum": 0, "nesterov":False }

N=10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = Model(D_in, specs, 10)
model.set_loss(criterion)
model.set_optimizer(name=optimizer, options=optspecs)
cmodel = model.coremdl
print(cmodel.parameters())


# tizio =  Model(D_in, specs, 10).coremdl #copy.deepcopy(cmodel)
# tizio.load_state_dict(cmodel.state_dict())
tizio = copy.deepcopy(cmodel)

print("PRE")
print(cmodel(x))
print(tizio(x))
model.single_update(x,y)
print("AFTER")
print(cmodel(x))
print(tizio(x))



