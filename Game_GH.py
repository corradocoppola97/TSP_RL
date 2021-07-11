from enum import Enum
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class TableType(Enum):
    random_graph = 1
    random_tree = 2
    random_table = 3
    random_jobs = 4

class RandomGraphSpecs(Enum):
    Nnodes = 1
    Nedges = 2
    Probability = 3
    Seed = 4
    Repetitions = 5
    Distribution = 6
    DistParams = 7

class RandomTreeSpecs(Enum):
    Depth = 1
    Seed = 2
    Repetitions = 3
    Distribution = 4
    DistParams = 5

class RandomTableSpecs(Enum):
    Njobs = 1
    Nmachines = 2
    Probability = 3
    Seed = 4
    Repetitions = 5
    Distribution = 6
    DistParams = 7

class RandomJobsSpecs(Enum):
    Njobs = 1
    Nmachines = 2
    Probability = 3
    Seed = 4
    Repetitions = 5
    Distribution = 6
    DistParams = 7

class gametable():
    def table(type=TableType.random_table, specs=None):
        if type == TableType.random_graph:
            return gametable._random_graph(specs)
        elif type == TableType.random_tree:
            return gametable._random_tree(specs)
        elif type == TableType.random_table:
            return gametable._random_table(specs)
        elif type == TableType.random_jobs:
            return gametable._random_jobs(specs)

    def _random_jobs(specs=None):
        njobs = specs[RandomJobsSpecs.Njobs]
        nmachines = specs[RandomJobsSpecs.Nmachines]
        pr = specs[RandomJobsSpecs.Probability]
        seed = specs[RandomJobsSpecs.Seed]
        repetitions = specs.get(RandomJobsSpecs.Repetitions)
        distribution = specs.get(RandomJobsSpecs.Distribution)
        distparams = specs.get(RandomJobsSpecs.DistParams)
        if repetitions is None:
            repetitions = 1
        if distribution is None:
            distribution = random.randint
        if distparams is None:
            distparams = {"lb": 1, "ub": 10}
        random.seed(a=seed)

        jobs=[[] for _ in range(njobs)]
        costs = [[{} for i in range(njobs)] for _ in range(repetitions)]

        machines=[i+1 for i in range(nmachines)]
        for j in range(njobs):
            machs=random.sample(machines,len(machines))
            for m in machs:
                chance = random.random()
                if chance <= pr:
                    jobs[j].append(m)

                    for rep in range(repetitions):
                        cost = gametable.custrand(distribution, distparams)
                        costs[rep][j][m] = round(cost,2) + 0.0

        return jobs, costs

    def _random_table(specs=None):
        njobs = specs[RandomTableSpecs.Njobs]
        nmachines = specs[RandomTableSpecs.Nmachines]
        pr = specs[RandomTableSpecs.Probability]
        seed = specs[RandomTableSpecs.Seed]
        repetitions = specs.get(RandomTableSpecs.Repetitions)
        distribution = specs.get(RandomTableSpecs.Distribution)
        distparams = specs.get(RandomTableSpecs.DistParams)
        if repetitions is None:
            repetitions = 1
        if distribution is None:
            distribution = random.randint
        if distparams is None:
            distparams = {"lb": 1, "ub": 10}
        random.seed(a=seed)

        operations=[]
        costs = [{} for _ in range(repetitions)]

        machines=[i for i in range(nmachines)]
        for i in range(njobs):
            machs=random.sample(machines,len(machines))
            for j in machs:
                chance = random.random()
                if chance <= pr:
                    op = (i, j)
                    operations.append(op)

                    for rep in range(repetitions):
                        cost = gametable.custrand(distribution, distparams)
                        costs[rep][op] = round(cost,2) + 0.0

        return operations, costs

    def custrand(distribution, distparams):
        if distribution == random.uniform or distribution == random.randint:
            return distribution(a=distparams["a"], b=distparams["b"])
        elif distribution == random.gauss:
            return max(0.1, distribution(mu=distparams["mu"], sigma=distparams["sigma"]))

    def _random_graph(specs=None):
        nnodes = specs[RandomGraphSpecs.Nnodes]
        nedges = specs[RandomGraphSpecs.Nedges]
        pr = specs[RandomGraphSpecs.Probability]
        seed = specs[RandomGraphSpecs.Seed]
        repetitions = specs.get(RandomGraphSpecs.Repetitions)
        distribution = specs.get(RandomGraphSpecs.Distribution)
        distparams = specs.get(RandomGraphSpecs.DistParams)
        if repetitions is None:
            repetitions = 1
        if distribution is None:
            distribution = random.randint
        if distparams is None:
            distparams = {"lb": 1, "ub": 10}
        random.seed(a=seed)

        edges = []
        costs = [{} for _ in range(repetitions)]
        count = 0

        step = 4

        interval = [set(range(i, min(i + step, nnodes))) for i in range(0, nnodes, step)]
        #print("Interval", interval)
        lenint = len(interval)
        for vv in range(lenint):
            pre = set([])
            actual = interval[vv]
            post = set([])
            if vv > 0:
                pre = interval[vv - 1]
            if vv < lenint - 1:
                post = interval[vv + 1]
            valset = set(list(pre) + list(actual) + list(post))
            for i in actual:
                for j in valset:
                    if i != j:
                        chance = random.random()
                        if chance <= pr:
                            edge = (i, j)
                            edges.append(edge)

                            for rep in range(repetitions):
                                cost = gametable.custrand(distribution, distparams)
                                costs[rep][edge] = cost + 0.0 + math.pow( i*len(valset)*0.02/nnodes + j*0.03/nnodes+gametable.custrand(distribution, distparams),2)
                            count += 1

        return edges, costs

    def custrand(distribution, distparams):
        if distribution == random.uniform or distribution == random.randint:
            return distribution(a=distparams["a"], b=distparams["b"])
        elif distribution == random.gauss:
            return max(0.1, distribution(mu=distparams["mu"], sigma=distparams["sigma"]))

    def _random_tree(specs=None):
        depth = specs[RandomTreeSpecs.Depth]
        seed0 = specs[RandomTreeSpecs.Seed]
        repetitions = specs.get(RandomTreeSpecs.Repetitions)
        distribution = specs.get(RandomTreeSpecs.Distribution)
        distparams = specs.get(RandomTreeSpecs.DistParams)
        if repetitions is None:
            repetitions = 1
        if distribution is None:
            distribution = random.randint
        if distparams is None:
            distparams = {"lb": 1, "ub": 10}
        #random.seed(a=seed0)

        edges = []
        costs = [{} for _ in range(repetitions)]

        for i in range(depth-1):
            start = sum(pow(2,j) for j in range(i))
            end = sum(pow(2,j) for j in range(i+1))
            for j in range(start, end):
                edges.append((j, 2 * j + 1))
                edges.append((j, 2 * j + 2))
                #edges.append((2 * j + 1, j))
                #edges.append((2 * j + 2, j))
                #edges.append((2 * j + 2,2 * j + 1))
                #edges.append((2 * j + 1, 2 * j + 2))
                for rep in range(repetitions):
                    costA = gametable.custrand(distribution, distparams)
                    costs[rep][(j, 2 * j + 1)] = costA + 0.0
                    #costs[rep][(2 * j + 1, j)] = costA + 0.0

                    costB = gametable.custrand(distribution, distparams)
                    costs[rep][(j, 2 * j + 2)] = costB + 0.0
                    #costs[rep][(2 * j + 2, j)] = costB + 0.0

                    #costs[rep][(2 * j + 2, 2 * j + 1)] = costA + 0.0
                    #costs[rep][(2 * j + 1, 2 * j + 2)] = costB + 0.0


        i = i+1
        start = sum(pow(2, j) for j in range(i))
        end = sum(pow(2, j) for j in range(i + 1))
        for j in range(start, end):
            edges.append((j, end))
            for rep in range(repetitions):
                costs[rep][(j, end)] = 1.0



        return edges, costs

    def _random_graph_distances(specs=None):
        nnodes = specs[RandomGraphSpecs.Nnodes]
        nedges = specs[RandomGraphSpecs.Nedges]
        pr = specs[RandomGraphSpecs.Probability]
        seed = specs[RandomGraphSpecs.Seed]
        repetitions = specs.get(RandomGraphSpecs.Repetitions)
        distribution = specs.get(RandomGraphSpecs.Distribution)
        distparams = specs.get(RandomGraphSpecs.DistParams)
        if repetitions is None:
            repetitions = 1
        if distribution is None:
            distribution = np.random.uniform
        if distparams is None:
            lb = 1
            ub = 100
            size = 2
        else:
            lb,ub,size = distparams #Da generalizzare!!!
        random.seed(a=seed)

        edges = []
        costs = [{} for _ in range(repetitions)]
        dist_matrixes = []
        #plt.figure()
        #plt.xlim(0, 100)
        #plt.ylim(0, 100)

        for rep in range(repetitions):
            v = []
            for _ in range(nnodes):
                delta = random.randint(0,1)
                #p = np.random.uniform(_,_,size=(2,))+np.random.normal(0,10,size=(2,))
                #p = abs(p)
                #p = np.random.normal(50,5,size=(2,))
                #p = delta*np.random.uniform(0,20,size=(2,))+(1-delta)*np.random.uniform(80,100,size=(2,)) + np.random.normal(0,10)
                #p = abs(p)
                p = np.random.uniform(0,100,size=(2,))
                v.append(p)
                #if rep==0:
                    #if _==0:
                        #plt.plot(p[0],p[1],'r.',markersize=5)
                    #else:
                        #plt.plot(p[0], p[1],'b.',markersize=5)

            #plt.show()
            dist_matrix = []
            for _ in range(nnodes):
                rr = []
                for __ in range(nnodes):
                    rr.append(0)
                dist_matrix.append(rr)

            for i in range(nnodes):
                for j in range(nnodes):
                    if i!=j:
                        dist_matrix[i][j] = int(np.linalg.norm(v[i]-v[j])) + 1
                        edge = (i,j)

                        if rep == 0:
                            edges.append(edge)

                        costs[rep][edge] = dist_matrix[i][j]
                    else:
                        dist_matrix[i][j] = 0
            dist_matrixes.append(dist_matrix)

        return edges,costs,dist_matrixes