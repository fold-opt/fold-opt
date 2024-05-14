import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from qpth.qp import QPFunction
import time


import sys
from multilabel_example.qpth_plus.qpth.qp import QPFunction_Plus
from casadi_solver import casadi_solver, dict_conv, to_cost_dict
from fold_opt.fold_opt import FoldOptLayer



####################################
#          SQP Functions           #
####################################

def grad_f(problem, y, cost, device=torch.device("cpu")):
    cost = cost.reshape(1, 3, 7)
    pg = y[problem.yindices[2]:problem.yindices[3]].to(device)
    grad = (((2 * (pg * cost[:, 0].flatten()) + cost[:, 0].flatten())) / problem.acopf_problem.obj_scaler).flatten()
    full_grad = torch.cat([torch.zeros(problem.yindices[2]).to(device), grad.to(device), torch.zeros(problem.yindices[-1]-problem.yindices[3]).to(device)])
    return full_grad


def g(problem, y, device=torch.device("cpu")):

    ## Extract elements from y ##

    vm = y[problem.yindices[1]:problem.yindices[2]].unsqueeze(0).to(device)
    dva = y[problem.yindices[-2]:problem.yindices[-1]].unsqueeze(0).to(device)
    

    ## Compute flow ##

    flow = problem.acopf_problem.compute_torch_flow(vm.cpu(), dva.cpu())

    ## Calculate ineq_resid ##

    pf_fr = flow["pf_fr"].to(device); qf_fr = flow["qf_fr"].to(device)
    pf_to = flow["pf_to"].to(device); qf_to = flow["qf_to"].to(device)
    tl_fr = pf_fr**2 + qf_fr**2 - torch.from_numpy(problem.acopf_problem.thermal_limit).to(device)**2
    tl_to = pf_to**2 + qf_to**2 - torch.from_numpy(problem.acopf_problem.thermal_limit).to(device)**2
    ineq = torch.cat([tl_fr,tl_to],axis=1)
    ineq = torch.clamp(ineq, 0.)

    return ineq.to(device)


def e(problem, y, c, device=torch.device("cpu")):

    # ## Extract elements from y ##
    pd_bus = torch.from_numpy(problem.x[0, problem.xindices[3]:problem.xindices[4]]).to(device)
    qd_bus = torch.from_numpy(problem.x[0, problem.xindices[4]:problem.xindices[5]]).to(device)

    vm = y[problem.yindices[1]:problem.yindices[2]].unsqueeze(0).to(device)
    dva = y[problem.yindices[-2]:problem.yindices[-1]].unsqueeze(0).to(device)
    
    ## Compute flow ##
    flow = problem.acopf_problem.compute_torch_flow(vm.cpu(), dva.cpu())


    pg = y[problem.yindices[2]:problem.yindices[3]].unsqueeze(0).to(device)
    qg = y[problem.yindices[3]:problem.yindices[4]].unsqueeze(0).to(device)

    pg_pad = problem.torch_pad(pg)
    pg_bus = pg_pad[:,problem.acopf_problem.bus_genidxs].sum(dim=2)
    qg_pad = problem.torch_pad(qg)
    qg_bus = qg_pad[:,problem.acopf_problem.bus_genidxs].sum(dim=2)


    ## Calculate eq_resid ##

    detach = False

    if detach:
        pf_fr_bus = flow["pf_fr_bus"].detach().to(device); pf_to_bus = flow["pf_to_bus"].detach().to(device)
        qf_fr_bus = flow["qf_fr_bus"].detach().to(device); qf_to_bus = flow["qf_to_bus"].detach().to(device)  
    else:   
        pf_fr_bus = flow["pf_fr_bus"].to(device); pf_to_bus = flow["pf_to_bus"].to(device)
        qf_fr_bus = flow["qf_fr_bus"].to(device); qf_to_bus = flow["qf_to_bus"].to(device)

    balance_p = pg_bus - pd_bus - pf_to_bus - pf_fr_bus - torch.from_numpy(problem.acopf_problem.gs).to(device)*vm**2
    balance_q = qg_bus - qd_bus - qf_to_bus - qf_fr_bus + torch.from_numpy(problem.acopf_problem.bs).to(device)*vm**2

    eq = torch.cat([balance_p.t(),balance_q.t()])
    
    return eq.to(device)


def lagrangian_first_order(problem, x, y, lam, nu, c_hat, device=torch.device("cpu")):
    grad_f_value = grad_f(problem, y, c_hat).to(device)
    grad_g_value = torch.func.jacrev(lambda y:  g(problem, y))(y).squeeze(0)
    grad_e_value = torch.func.jacrev(lambda y:  e(problem, y, x))(y).squeeze(1)

    ineq_grad = torch.sum(nu.unsqueeze(1) * grad_g_value, dim=0).to(device)
    eq_grad = torch.sum(lam.unsqueeze(1) * grad_e_value, dim=0).to(device)
    return (grad_f_value + ineq_grad + eq_grad)


def hess_L(problem, x, y, lam, nu, c_hat, approx=False, eps=1e-3, device=torch.device("cpu")):

    _, lagrangian_second_order = torch.func.vjp(lambda solution: lagrangian_first_order(problem, x, solution, lam, nu, c_hat), y)

    hessian = torch.zeros(len(y), len(y))

    for i in range(len(y)):

        grad_output = torch.zeros(len(y), dtype=torch.float64).to(device)
        grad_output[i] = 1.0

        retain = True
        hessian_grad = lagrangian_second_order(grad_output)[0]
        hessian[i] = hessian_grad
        
        # Add regularization term to make matrix SPD
        hessian[i,i] += 1

    return hessian



""" COPY THESE LINES """""""""""""""""""""""""""""

def get_projection_layer(A,b,G,h):
    N = A.shape[1] # number of variables  
    Q = torch.eye(N)
    return lambda x: QPFunction(verbose=-1,eps=1e-10,maxIter=1000)(2*Q.double(),-2*x.double(),G.double(),h.double(),A.double(),b.double())   


def get_sqp_layer(g, e, grad_g, grad_e, grad_f, hess_L):
    
    def sqp_layer(x,c,c_hat,lam,nu_d,form):
        G = torch.stack( [grad_g(x_i,form).double() for x_i in x] ).squeeze(0)
        h = torch.stack( [-g(x_i,form).double() for x_i in x] ).squeeze(1)

        A = torch.stack( [grad_e(x_i,c[i],form)[0].double() for i,x_i in enumerate(x)] ).squeeze(2)
        b = torch.stack( [-e(x_i,c[i],form).double() for i,x_i in enumerate(x)] ).squeeze(-1)

        Q = torch.stack( [hess_L(x[i],lam[i],nu_d[i],torch.from_numpy(form.y),c_hat,form).double()    for i in range(len(x))] )
        p = torch.stack( [grad_f(x_i,c_hat,form).double() for i,x_i in enumerate(x)] )

        #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically
        primaldual = QPFunction_Plus(verbose=-1,eps=1e-14,maxIter=1000)(Q,p,G,h,A,b)
        return primaldual
    
    return sqp_layer


def SQP(x0,c_hat,c,sqp,N,M_in,M_eq,alpha,n_iter=1):
    xi = x0[:,:N]
    lam = x0[:, N+M_eq:]
    nu_d = x0[:, N:N+M_eq]
    for i in range(n_iter):
        primaldual = sqp(xi,c[0],c_hat,lam,nu_d,c[1])
        d          = primaldual[ :,:N ]
        lam_d      = primaldual[ :, N:N+M_in ]
        nu_d       = primaldual[ :,   N+M_in:N+M_in+M_eq ]
        xi         = xi+alpha*d
        lam        = lam + alpha*(lam_d - lam)
    return torch.cat([xi,lam,nu_d], dim=1)

""""""""""""""""""""""""""""""""""""""""""""""""



class LayerSQP():
    def __init__(self, N, C, acopf_problem, stepsize=1.0, n_iter=1):
        super().__init__()

        self.alpha = stepsize
        eps = 1.0
        
        self.N = N
        self.M_in = 114
        self.M_eq = 160

        
        self.grad_f   = lambda x, c, form:                 grad_f(form, x, c)
        self.hess_f   = lambda y, lam, nu, x, c, form:     hess_L(form, x, y, lam, nu, c)
        self.g        = lambda x, form:                    g(form, x)
        self.e        = lambda x, c, form:                 e(form, x, c)

        self.grad_g   = torch.func.jacrev(self.g)
        self.grad_e   = torch.func.jacrev(self.e, argnums=(0,1))

        
        self.sqp = get_sqp_layer(self.g,self.e,self.grad_g,self.grad_e,self.grad_f,self.hess_f)
        
        solver = casadi_solver(acopf_problem, C)
        self.fwd_solver_batch = lambda c: solver.solve(acopf_problem.trainX, acopf_problem.trainY, c)[0]
        
        self.update_step  = lambda c_hat,x,c: SQP(x,c_hat,c,self.sqp,self.N,self.M_in,self.M_eq,self.alpha,n_iter=1).float()
        self.fold_opt_layer = FoldOptLayer(self.fwd_solver_batch, self.update_step, n_iter=n_iter, backprop_rule='FPI')

    def solve(self, c_hat, c):
        return self.fold_opt_layer( c_hat, ground=c )
        