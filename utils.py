from enum import Enum
import random

class TableType(Enum):
    random_graph = 1

class RandomGraphSpecs(Enum):
    Nnodes = 1
    Nedges = 2
    Probability = 3
    Seed = 4
    Repetitions = 5
    Distribution = 6
    DistParams = 7


class gametable():
    def table(type = TableType.random_graph, specs = None):
        if type == TableType.random_graph:
            return gametable._random_graph(specs)
    def _random_graph(specs = None):
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
            distparams = { "lb": 1, "ub":10}
        random.seed(a=seed)

        edges = []
        costs = [{} for rep in range(repetitions)]
        count = 0

        step = 4

        def custrand(distribution, distparams):
            if distribution == random.randint:
                return distribution(lb=distparams["lb"], ub=distparams["ub"])
            elif distribution == random.uniform:
                return distribution(a=distparams["a"], b=distparams["b"])
            elif distribution == random.gauss:
                return max(0.1, distribution(mu=distparams["mu"], sigma=distparams["sigma"]))




        interval = [set(range(i, min(i+step, nnodes))) for i in range(0,nnodes, step)]
        print("Interval", interval)
        lenint = len(interval)
        for  vv in range(lenint):
            pre = set([])
            actual = interval[vv]
            post = set([])
            if vv > 0:
                pre = interval[vv - 1]
            if vv < lenint - 1:
                post = interval[vv + 1]
            valset = set( list(pre) + list(actual) + list(post))
            for i in actual:
                for j in valset:
                    if i != j:
                        chance = random.random()
                        if chance <= pr:
                            edge = (i, j)
                            edges.append(edge)

                            for rep in repetitions:
                                costs[rep][edge] = custrand(distribution, distparams) + 0.0
                            count += 1

        return edges, costs





"""
D_out = nedges
D_in = nedges + nedges
specs2 = [
    ("sigmoid", nedges*2),
    ("relu", nnodes),
    #("logsigmoid", nedges),
    ("linear", D_out)
]

thoralgo = thor2(environment_specs,
                 D_in,
                 specs2,
                 criterion,
                 optimizer,
                 optspecs,
                 memorylength,
                 memorypath,
                 seed,
                 stop_function)

nepisodes = 1
stats = thoralgo.solve(nepisodes = nepisodes,
               display = (True, 10),
               randomness0 = 1,
               batchsize = 10,
               maximumiter = nnodes,
               steps = 4
               )



#print(stats)
final_objectives = [stat["final_objective"] for stat in stats if stat["is_final"] == 1]
basevals = [-baseval for i in range(len( final_objectives))]

plt.plot(final_objectives)
plt.plot(basevals)
plt.show()
plt.cla()
#
# cumulatives = [stat["cumulative_reward"] for stat in stats if stat["is_final"] == 1]
# maxlen = 0
#
# for i in range(0,len(cumulatives), math.ceil(nepisodes/20)):
#     plt.plot(cumulatives[i])
#     if len(cumulatives[i]) > maxlen:
#         maxlen = len(cumulatives[i])
# basevals = [-baseval for i in range(maxlen)]
# plt.plot(basevals)
# plt.show()



bestsol = -1e10
bestsolpath = []

bestsol2 = bestsol
bestsolpath2 = []
for stat in stats:
    if -baseval + 1 >= stat["final_objective"] >= bestsol:
        bestsol = stat["final_objective"]
        bestsolpath = stat["solution"]
    if  stat["final_objective"] >= bestsol2:
        bestsol2 = stat["final_objective"]
        bestsolpath2 = stat["solution"]
print("    BASELINE", basesol,      " BASELINE", -baseval)
print("BESTSOL PATH", bestsolpath,  "  BESTSOL", bestsol)
print("BESTSOL2PATH", bestsolpath2, " BESTSOL2", bestsol2)
print("LASTSOL2PATH", stats[len(stats)-1]["solution"], " LASTSOL2", stats[len(stats)-1]["final_objective"])
print("edges",edges)
print("costs",costs)
"""




