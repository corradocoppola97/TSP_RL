#from docplex.mp.model import Model, Context
#from docplex.mp.solution import SolveSolution
import time
import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


# f = pd.read_fwf(r'C:\Users\Marta\Desktop\TESI\abz5.txt', header=None, sep='  ')
# t = np.array(f)
# table = [[int(t[i][j]) for j in range(0, t.shape[1], 2) if not np.isnan(t[i][j])] for i in range(t.shape[0])]
# times=[{t[i][j]: t[i][j+1] for j in range(0, t.shape[1], 2) if not np.isnan(t[i][j])} for i in range(len(table))]

class baseline():
    def __init__(self):
        self.cpmod = Model()

    def js_jobs(self, jobs, proc_times):

        operations = [(i,j) for i in range(len(jobs)) for j in jobs[i]]
        proc_times = {(i,j): proc_times[i][j] for (i,j) in operations}

        jobs = set([(i) for (i, j) in operations])  #insieme dei job
        machines = set([(j) for (i, j) in operations])  #insieme delle macchine

        tij = {(i, j): self.cpmod.continuous_var(name="t" + str(i) + "_" + str(j), lb=0) for (i, j) in operations}  #start time of job i on machine i, that is start time of operation

        xijk = {(i, j, k): self.cpmod.binary_var(name="x" + str(i) + "_" + str(j) + "_" + str(k))
                for i in jobs for j in jobs if j>i for k in machines if (i,k) in operations and (j,k) in operations} #binary var, 1 if job i precedes job j on machine k, 0 otherwise

        C_max=self.cpmod.continuous_var(name="C_max",lb=0)  #tempo di completamento massimo

        obj = self.cpmod.minimize(C_max)  #min makespan

        cons_C = [self.cpmod.add_constraint(C_max >= tij[op] + proc_times[op]) for op in operations]
        #cons_C = [self.cpmod.add_constraint(C_max >= tij[operations[i]] + proc_times[operations[i]]) for i in range(len(operations)) if i == len(operations)-1 or operations[i][0] != operations[i+1][0]]

        cons_prec = [self.cpmod.add_constraint(
            tij[operations[i]] - tij[operations[i - 1]] >= proc_times[operations[i - 1]])
            for i in range(1, len(operations)) if operations[i][0] == operations[i - 1][0]
        ]   #vincoli di precedenza

        M=sum(proc_times.values())+1

        cons_disj1 = [self.cpmod.add_constraint(
            tij[(i,k)] + proc_times[(i,k)] <= tij[(j,k)] + M * (1 - xijk[i,j,k]))
            for i in jobs for j in jobs if j>i for k in machines if (i,k) in operations and (j,k) in operations]  #vincoli disgiuntivi 1

        cons_disj2 = [self.cpmod.add_constraint(
            tij[(j,k)] + proc_times[(j,k)] <= tij[(i,k)] + M * xijk[i, j, k])
            for i in jobs for j in jobs if j>i for k in machines if (i,k) in operations and (j,k) in operations]  #vincoli disgiuntivi 2

        start = time.time()
        #self.cpmod.parameters.mip.tolerances.mipgap = 5e-2
        #self.cpmod.set_time_limit(300)
        self.cpmod.solve(log_output=False)
        end = time.time()

        #print(self.cpmod.export_as_lp_string())
        sol = {op:tij[op].solution_value for op in operations} #ordino in base ai tempi?
        #arrotondo tempi di soluzione?

        return self.cpmod.objective_value, sol, end - start

    def job_shop_scheduling(self, operations, proc_times):

        jobs = set([(i) for (i, j) in operations])  #insieme dei job
        machines = set([(j) for (i, j) in operations])  #insieme delle macchine

        tij = {(i, j): self.cpmod.continuous_var(name="t" + str(i) + "_" + str(j), lb=0) for (i, j) in operations}  #start time of job i on machine i, that is start time of operation

        xijk = {(i, j, k): self.cpmod.binary_var(name="x" + str(i) + "_" + str(j) + "_" + str(k))
                for i in jobs for j in jobs if j>i for k in machines if (i,k) in operations and (j,k) in operations} #binary var, 1 if job i precedes job j on machine k, 0 otherwise

        C_max=self.cpmod.continuous_var(name="C_max",lb=0)  #tempo di completamento massimo

        obj = self.cpmod.minimize(C_max)  #min makespan

        cons_C = [self.cpmod.add_constraint(C_max >= tij[op] + proc_times[op]) for op in operations]
        #cons_C = [self.cpmod.add_constraint(C_max >= tij[operations[i]] + proc_times[operations[i]]) for i in range(len(operations)) if i == len(operations)-1 or operations[i][0] != operations[i+1][0]]

        cons_prec = [self.cpmod.add_constraint(
            tij[operations[i]] - tij[operations[i - 1]] >= proc_times[operations[i - 1]])
            for i in range(1, len(operations)) if operations[i][0] == operations[i - 1][0]
        ]   #vincoli di precedenza

        M=sum(proc_times.values())+1

        cons_disj1 = [self.cpmod.add_constraint(
            tij[(i,k)] + proc_times[(i,k)] <= tij[(j,k)] + M * (1 - xijk[i,j,k]))
            for i in jobs for j in jobs if j>i for k in machines if (i,k) in operations and (j,k) in operations]  #vincoli disgiuntivi 1

        cons_disj2 = [self.cpmod.add_constraint(
            tij[(j,k)] + proc_times[(j,k)] <= tij[(i,k)] + M * xijk[i, j, k])
            for i in jobs for j in jobs if j>i for k in machines if (i,k) in operations and (j,k) in operations]  #vincoli disgiuntivi 2

        start = time.time()
        #self.cpmod.parameters.mip.tolerances.mipgap = 5e-2
        #self.cpmod.set_time_limit(300)
        self.cpmod.solve(log_output=True)
        end = time.time()

        #print(self.cpmod.export_as_lp_string())
        sol = {op:tij[op].solution_value for op in operations} #ordino in base ai tempi?
        #arrotondo tempi di soluzione?

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

class baseline_ortools():
    def __init__(self,dis_m,n):
        D = []
        m = len(dis_m)
        for j in range(m):
            dataset = [[0 for _ in range(n)] for __ in range(n)]
            costs = dis_m[j]
            for edge in costs:
                i,j = edge
                dataset[i][j] = costs[edge]
                dataset[j][i] = costs[edge]
            D.append(dataset)
        self.dataset = D



    def risolvi_problema(self,n, data):
        #print('data',data)
        data = {'distance_matrix':data,'num_vehicles':1,'depot':0}
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


    def print_solution(self,manager, routing, solution):
        #print('f(x*): {}'.format(solution.ObjectiveValue()))
        ind = routing.Start(0)
        output = 'Routing per il veicolo 0:\n'
        lunghezza_ciclo = 0
        while not routing.IsEnd(ind):
            output += ' {} -> '.format(manager.IndexToNode(ind))
            prev_ind = ind
            ind = solution.Value(routing.NextVar(ind))
            lunghezza_ciclo += routing.GetArcCostForVehicle(prev_ind, ind, 0)

        output += ' {}\n'.format(manager.IndexToNode(ind))
        #print(output)
        # print('Lunghezza totale percorsa: {} \n'.format(lunghezza_ciclo))
        return output, lunghezza_ciclo

    def solve_ortools(self,printFLAG=False):
        ortools_values = []
        comp_time = []
        outputs = []
        for i in range(len(self.dataset)):
            dm = self.dataset[i]
            n = len(dm)
            t1 = time.time()
            sol, man, rout = self.risolvi_problema(n,dm)
            t2 = time.time()
            opt_time = t2-t1
            comp_time.append(opt_time)
            ortools_values.append(sol.ObjectiveValue())
            if printFLAG:
                out, length = self.print_solution(man,rout,sol)
                outputs.append((out,length))

        return ortools_values,comp_time,outputs

    def TestPlot(self,ortools_values, comp_time, test_stats):
        times_TSP = []
        values_TSP = []
        n_test = len(test_stats)
        n_tot = len(ortools_values)
        for i in range(len(test_stats)):
            times_TSP.append(test_stats[i]['time'])
            values_TSP.append(test_stats[i]['final_objective'])
        t, v = np.array(times_TSP, dtype=float), np.array(values_TSP, dtype=float)
        v_or, t_or = np.array(ortools_values[n_tot - n_test:], dtype=float), np.array(comp_time[n_tot - n_test],
            dtype=float)
        time_ratio = t / t_or
        value_ratio = -v / v_or
        return time_ratio, value_ratio








