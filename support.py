from docplex.mp.model import Model, Context
from docplex.mp.solution import SolveSolution
import time
import numpy as np

class baseline():
    def __init__(self):
        self.cpmod = Model()

    def job_shop_scheduling(self, table):
        operations = []    #mi creo insieme di operazioni: tuple (job, macchina)
        for i in range(len(table)):
            job = i
            for j in range(len(table[i])):
                mach = table[i][j][0]
                op = (job, mach)
                operations.append(op)

        proc_time = {}  #dizionario dei tempi di processamento: chiave = operazione : valore=processing time
        for i in range(len(table)):
            job = i
            for j in range(len(table[i])):
                mach = table[i][j][0]
                op = tuple((job, mach))
                proc_time[op] = table[i][j][1]

        jobs = set([(i) for (i, j) in operations])  #insieme dei job
        machines = set([(j) for (i, j) in operations])  #insieme delle macchine

        tij = {(i, j): self.cpmod.continuous_var(name="t" + str(i) + "_" + str(j), lb=0) for (i, j) in operations}  #start time of job i on machine i, that is start time of operations
        xijk = {(i, j, k): self.cpmod.binary_var(name="x" + str(i) + "_" + str(j) + "_" + str(k))
                for i in jobs for j in jobs if j>i for k in machines if (i,k) in operations and (j,k) in operations} #binary var, 1 if job i precedes job j on machine k, 0 otherwise

        C_max=self.cpmod.continuous_var(name="C_max",lb=0)  #tempo di completamento massimo
        cons_C=[self.cpmod.add_constraint(C_max >= tij[op] + proc_time[op]) for op in operations]
        obj = self.cpmod.minimize(C_max)  #min makespan

        cons_prec = [self.cpmod.add_constraint(
            tij[operations[i]] - tij[operations[i - 1]] >= proc_time[operations[i - 1]]) for i in
            range(1, len(operations)) if operations[i][0] == operations[i - 1][0]
        ]   #vincoli di precedenza

        M = 1e+6

        cons_disj1 = [self.cpmod.add_constraint(
            tij[operations[i]] + proc_time[operations[i]] <= tij[operations[j]] + M * (
                        1 - xijk[(operations[i][0], operations[j][0], operations[i][1])])) for i in
            range(len(operations)) for j in range(len(operations)) if j > i and operations[i][1] == operations[j][1]
        ]  #vincoli disgiuntivi 1

        cons_disj2 = [self.cpmod.add_constraint(
            tij[operations[j]] + proc_time[operations[j]] <= tij[operations[i]] + M * (
            xijk[(operations[i][0], operations[j][0], operations[i][1])])) for i in range(len(operations)) for j in
            range(len(operations)) if j > i and operations[i][1] == operations[j][1]
        ] #vincoli disgiuntivi 2

        start = time.time()
        self.cpmod.solve()
        end = time.time()

        #print(self.cpmod.export_as_lp_string())
        sol = []
        ord = sorted(tij.items())
        for i in ord:
            sol.append(i[0])

        return self.cpmod.objective_value, sol, end - start

    def min_path(self, nnodes, edges, costs):
        xij = {(i,j) : self.cpmod.binary_var(name= "x"+str(i)+"_"+str(j)) for (i,j) in edges}

        obj = self.cpmod.minimize(
            self.cpmod.sum( xij[edge]*costs[edge] for edge in edges)
        )

        cons = [ self.cpmod.add_constraint(
            self.cpmod.sum( xij[edge] for edge in edges if edge[1] == node) ==
            self.cpmod.sum(xij[edge] for edge in edges if edge[0] == node)
        )   for node in range(1, nnodes - 1)]

        start =  self.cpmod.add_constraint(
            self.cpmod.sum( xij[edge] for edge in edges if edge[1] == 0) ==
            self.cpmod.sum(xij[edge] for edge in edges if edge[0] == 0) - 1
        )

        end =  self.cpmod.add_constraint(
            self.cpmod.sum( xij[edge] for edge in edges if edge[1] == nnodes - 1) ==
            self.cpmod.sum(xij[edge] for edge in edges if edge[0] == nnodes - 1) + 1
        )


        start = time.time()
        self.cpmod.solve()
        end = time.time()

        nextnode = 0
        sol = []
        totarcs = sum(1 if xij[edge].solution_value >= 0.5 else 0 for edge in edges)
        count = 0
        while count != totarcs - 1:
            flag = False
            for (i,j) in edges:
                if xij[(i,j)].solution_value >= 0.5 and i == nextnode:
                    sol.append((i,j))
                    count += 1
                    nextnode = j
                    flag = True
                    break
            if not flag:
                break
        for (i, j) in edges:
            if xij[(i, j)].solution_value >= 0.5 and j == nnodes - 1:
                sol.append((i, j))
                count += 1
                break


        return self.cpmod.objective_value, sol, end-start

