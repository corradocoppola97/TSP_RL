from docplex.mp.model import Model, Context
from docplex.mp.solution import SolveSolution
import time

class baseline():
    def __init__(self):
        self.cpmod = Model()

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

