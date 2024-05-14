import helper_acopf
from helper_acopf import PDLDataSet
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from problem import problem, reconstruct_dict
import casadi as ca


class casadi_solver:

    def __init__(self, acopf_problem, batchsize):

        self.acopf_problem = acopf_problem
        self.batchsize = batchsize


    def solve(self, x, y, cost, full=False):


        if type(cost) is not dict:
            cost = {
                'quad_cost' :  cost[:, 0].flatten().cpu().numpy(),
                'lin_cost' :   cost[:, 1].flatten().cpu().numpy(),
                'const_cost' : cost[:, 2].flatten().cpu().numpy()
            }

        # Batch size, number of inequalities, number of equalities
        N = self.batchsize
        M_in = 114
        M_eq = 160

        equality_degree = 1e-8
        verbose = False

        coeffs_list = []
        primal_list = []
        dual_list = []

        for idx in range(N):        
            ## Convert problem from dict to tensor ##
            primal_dict = y
            primal_full = []
            indices = [0]
            N_var = 0
            for item in primal_dict.keys():
                indices.append(N_var+primal_dict[item].shape[1])
                N_var += primal_dict[item].shape[1]
                primal_full.append(primal_dict[item].transpose())

            primal = np.concatenate(primal_full).transpose()
            primal = primal[idx, :]
            primal = np.expand_dims(primal, axis=0)

            x_dict = x
            x_tensor = []
            xindices = [0]
            N_var = 0
            for item in x_dict.keys():
                xindices.append(N_var+x_dict[item].shape[1])
                N_var += x_dict[item].shape[1]
                x_tensor.append(x_dict[item].transpose())

            x_tensor = np.concatenate(x_tensor).transpose()
            x_tensor = x_tensor[idx, :]
            x_tensor = np.expand_dims(x_tensor, axis=0)



            ## Bounds ##

            bound_57 = np.ones_like(self.acopf_problem.vmmin)*10
            lb = ca.repmat(ca.DM(np.concatenate([-1*bound_57, self.acopf_problem.vmmin, self.acopf_problem.pgmin, self.acopf_problem.qgmin, self.acopf_problem.angmin])), 1, 1)
            ub = ca.repmat(ca.DM(np.concatenate([bound_57, self.acopf_problem.vmmax, self.acopf_problem.pgmax, self.acopf_problem.qgmax, self.acopf_problem.angmax])), 1, 1)
            cl = ca.repmat(ca.DM(np.concatenate([-equality_degree*np.ones(M_eq), -equality_degree*np.ones(M_in)])), 1, 1)
            cu = ca.repmat(ca.DM(np.concatenate([equality_degree*np.ones(M_eq), equality_degree*np.ones(M_in)])), 1, 1)

            ## Set-up for casadi variables ##

            ipopt_problem = problem(x_tensor, x_dict, primal, primal_dict, self.acopf_problem, indices, xindices, M_eq, M_in, cost, verbose=verbose)
            constraint_batch = lambda x: ipopt_problem.constraints_ca(x, idx)
            opt_gap = lambda y_val: self.acopf_problem.opt_gap_ca(x_tensor, y_val, idx, ipopt_problem.yindices, cost)

            primal0 = ca.reshape(primal, -1, 1)
            primal_solve = ca.MX.sym('x', 208, 1)
            constraint = ca.MX(constraint_batch(primal_solve))
            objective = ca.MX(opt_gap(primal_solve))

            opts = {"ipopt.max_iter": 100000, "ipopt.print_level": 0, "print_time": 0, "ipopt.constr_viol_tol": 1e-8, "ipopt.tol": 1e-8, "ipopt.mu_strategy": 'adaptive'}
            nlp = {'x': primal_solve, 'f': objective, 'g': constraint}
            solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

            result = solver(x0=primal0, lbx=lb, ubx=ub, lbg=cl, ubg=cu)
            x_sol = np.array(result['x']).flatten()

            coeffs_list.append(torch.tensor(np.reshape(np.stack(x_tensor), (1, x_tensor.shape[1]))))
            primal_list.append(torch.tensor(np.reshape(np.stack(x_sol), (1, primal.shape[1]))))
            dual_list.append(torch.from_numpy(np.array(result['lam_g']).flatten()).reshape(1, (M_eq + M_in)))


        if full:
            # Fixed Point Conditions
            coeffs = torch.stack(coeffs_list).squeeze(1)
            primal = torch.stack(primal_list).squeeze(1)
            dual = torch.stack(dual_list).squeeze(1)

            return coeffs, primal, dual, ipopt_problem


        primaldual = torch.cat([torch.stack(primal_list).squeeze(1), torch.stack(dual_list).squeeze(1)], dim=1)
        return primaldual, ipopt_problem

        

def dict_conv(x, y):

    primal_dict = y
    primal_full = []
    indices = [0]
    N_var = 0
    for item in primal_dict.keys():
        indices.append(N_var+primal_dict[item].shape[1])
        N_var += primal_dict[item].shape[1]
        primal_full.append(primal_dict[item].transpose())

    primal = np.concatenate(primal_full).transpose()
    primal = primal[0, :]
    primal = np.expand_dims(primal, axis=0)

    x_dict = x
    x_tensor = []
    xindices = [0]
    N_var = 0
    for item in x_dict.keys():
        xindices.append(N_var+x_dict[item].shape[1])
        N_var += x_dict[item].shape[1]
        x_tensor.append(x_dict[item].transpose())

    x_tensor = np.concatenate(x_tensor).transpose()
    x_tensor = x_tensor[0, :]
    x_tensor = np.expand_dims(x_tensor, axis=0)

    return x_tensor, x_dict, primal, primal_dict, indices, xindices


def to_cost_dict(c):
    return {
        'quad_cost':  c[0],
        'lin_cost':   c[1],
        'const_cost': c[2]
    }