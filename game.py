from enum import Enum
import random
import math


class TableType(Enum):
    random_graph = 1
    random_tree = 2


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


class gametable():
    def table(type=TableType.random_graph, specs=None):
        if type == TableType.random_graph:
            return gametable._random_graph(specs)
        elif type == TableType.random_tree:
            return gametable._random_tree(specs)

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

        step = 2

        interval = [set(range(i, min(i + step, nnodes))) for i in range(0, nnodes, step)]
        print("Interval", interval)
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
                                costs[rep][edge] = cost + 0.0 + math.pow( i*len(valset)*0.02 + j*0.03+gametable.custrand(distribution, distparams),2)
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







