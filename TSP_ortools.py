import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import torch
from sklearn.preprocessing import MinMaxScaler
from warnings import filterwarnings

filterwarnings('ignore')


def crea_problema(n, a, b):
    data = {}
    matrice = []
    for i in range(n):
        r = []
        for j in range(n):
            r.append(0)
        matrice.append(r)

    l = []
    for i in range(n):
        pos = np.random.uniform(a, b, 2)
        l.append(pos)

    for i in range(n):
        for j in range(n):
            if i!=j: matrice[i][j] = int(np.linalg.norm(l[i] - l[j])) + 1

    data['distance_matrix'] = matrice
    data['num_vehicles'] = 1
    data['depot'] = 0

    return data


def risolvi_problema(n, data):
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(da_i, a_j):
        da_nodo = manager.IndexToNode(da_i)
        a_nodo = manager.IndexToNode(a_j)
        return data['distance_matrix'][da_nodo][a_nodo]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    return solution, manager, routing


def print_solution(manager, routing, solution):
    print('f(x*): {}'.format(solution.ObjectiveValue()))
    ind = routing.Start(0)
    output = 'Routing per il veicolo 0:\n'
    lunghezza_ciclo = 0
    while not routing.IsEnd(ind):
        output += ' {} -> '.format(manager.IndexToNode(ind))
        prev_ind = ind
        ind = solution.Value(routing.NextVar(ind))
        lunghezza_ciclo += routing.GetArcCostForVehicle(prev_ind, ind, 0)

    output += ' {}\n'.format(manager.IndexToNode(ind))
    print(output)
    # print('Lunghezza totale percorsa: {} \n'.format(lunghezza_ciclo))
    return output, lunghezza_ciclo


def genera_Dataset(num_nodi, num_grafi):
    X = []
    Y = []
    dataset = []
    for j in range(num_grafi):
        # print('Creo problema {} di {}'.format(j+1,num_grafi))
        data_j = crea_problema(num_nodi,0,100)
        sol, man, rout = risolvi_problema(num_nodi, data_j)
        fob = float(sol.ObjectiveValue())
        mat_dist = data_j['distance_matrix']
        X.append(mat_dist)
        Y.append(fob)

    X, Y = np.array(X, dtype=float), np.array(Y, dtype=float)
    # X_res = np.reshape(X,(num_grafi,int(num_nodi*num_nodi)))
    # scaler = MinMaxScaler()
    # scaler.fit(X_res)
    # X_norm = scaler.transform(X_res)
    # X = np.reshape(X_norm,(num_grafi,num_nodi,num_nodi))
    for j in range(num_grafi):
        x = torch.from_numpy(X[j]).view(1, num_nodi, num_nodi)
        y = Y[j]
        campione = (x, y)
        dataset.append(campione)

    return dataset


# dataset = genera_Dataset(10,10)

def split(perc, dataset):
    n = len(dataset)
    n_test = int(perc * n)
    idx = np.random.permutation(n)
    train_ind = idx[n_test:]
    test_ind = idx[:n_test]
    trainset = [dataset[i] for i in train_ind]
    testset = [dataset[i] for i in test_ind]
    return trainset, testset


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_nodi):
        super(Net, self).__init__()
        self.n_nodi = n_nodi
        self.conv1 = nn.Conv2d(1, 10, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 10, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(10, 1, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(self.n_nodi, 30)
        self.fc2 = nn.Linear(30, 1)

    def forward(self, x, n_nodi):
        bs = x.shape[0]
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(bs,self.n_nodi,self.n_nodi)
        x = torch.mean(x,2)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        return x


def epoch(tl, optim, lossf, n_nodi):
    valori_loss = []
    for x, y in tl:
        x = x.float()
        y = y.float()
        out = net(x, n_nodi)
        loss = lossf(out, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        valori_loss.append(loss.item())

    avg_loss = sum(valori_loss) / len(valori_loss)
    return avg_loss


def train(epochs, tl, optim, lossf, n_nodi, test_loader):
    l_train = []
    l_test = []
    for j in range(epochs):
        loss = epoch(tl, optim, lossf, n_nodi)
        if (j) % 1==0:
            print('Epoch:{}   Loss:{:.4f}'.format(j, loss))
            loss_t, data = test_error(test_loader, lossf)
            print('Test_error: {}'.format(loss_t))
            l_train.append(loss)
            l_test.append(loss_t)
    return l_train, l_test, data


def test_error(test_loader, loss_fun):
    l = []
    data = []
    for x, y in test_loader:
        x, y = x.float(), y.float()
        out = net(x, 20)
        l.append(loss_fun(out, y).item())
        data.append((out.item(),y.item()))
    return sum(l) / len(l), data

n_nodi = 75
n_epochs = 25
bs = 32
dataset = torch.load("miao.pt")#genera_Dataset(n_nodi,1000)
trainset, testset = split(0.2,dataset)
trainloader = torch.utils.data.DataLoader(trainset,bs)
testloader = torch.utils.data.DataLoader(testset,1)
loss_fun = F.mse_loss
net = Net(n_nodi=n_nodi)
opt = torch.optim.Adam(net.parameters(),lr=1e-3)
l_train, l_test, data = train(n_epochs,trainloader,opt,loss_fun,n_nodi,testloader)

import matplotlib.pyplot as plt
l1,l2 = np.array(l_train,dtype=float), np.array(l_test,dtype=float)
plt.plot(l1)
plt.plot(l2)
plt.show()        
