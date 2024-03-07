

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from fold_opt.fold_opt import FoldOptLayer
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import cvxpy as cp

from multilabel_example.qpth_plus.qpth.qp import QPFunction
from multilabel_example.qpth_plus.qpth.qp import QPFunction_Plus



""" COPY THESE LINES """""""""""""""""""""""""""""
def obj_grad_fn(x,c,eps):
    return -c + eps*(torch.log(x) + 1.0 )

def fwd_solver(c, N, C, eps, primaldual=False):
    x = cp.Variable(N)
    constraints = [  0<=x, x<=1, cp.sum(x) == C  ]
    problem = cp.Problem(cp.Maximize(c @ x + eps*cp.sum(cp.entr(x))), constraints)

    solution = problem.solve(solver=cp.ECOS)
    primal   = x.value
    dual = np.concatenate( tuple([np.array([constr.dual_value]).flatten() for constr in constraints]) )
    
    if primaldual: return torch.cat([torch.Tensor(primal), torch.Tensor(dual)])
    return torch.Tensor(primal)


def get_projection_layer(A,b,G,h):
    N = A.shape[1] # number of variables  
    Q = torch.eye(N)
    return lambda x: QPFunction(verbose=-1,eps=1e-10,maxIter=1000)(2*Q.double(),-2*x.double(),G.double(),h.double(),A.double(),b.double())   

def get_sqp_layer(g, e, grad_g, grad_e, grad_f, hess_L):
    
    def sqp_layer(x,c,lam):
        G = torch.stack( [grad_g(x_i).double() for x_i in x] )
        h = torch.stack( [-g(x_i).double() for x_i in x] )
        if e != None:
            A = torch.stack( [grad_e(x_i).double() for x_i in x] )
            b = torch.stack( [-e(x_i).double() for x_i in x] )
        else:
            A = torch.Tensor([])
            b = torch.Tensor([])

        Q = torch.stack( [hess_L(x[i],lam[i],c[i] ).double()    for i in range(len(x))] )
        p = torch.stack( [grad_f(x_i,c[i]).double() for i,x_i in enumerate(x)] )

        #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically
        primaldual = QPFunction_Plus(verbose=-1,eps=1e-14,maxIter=1000)(Q,p,G,h,A,b)
        return primaldual
    
    return sqp_layer


# All data inputs are expected to be torch Tensors
def PGD(x0,c,projection,obj_grad,alpha=0.01,n_iter=1):
    xi = x0
    for i in range(n_iter):
        grads = obj_grad(xi,c)
        xi = projection(xi - alpha*grads)
    return xi


def SQP(x0,c,sqp,N,M_in,M_eq,alpha=0.01,n_iter=1):
    xi = x0[:,:N]
    lam = x0[:,N:N+M_in]
    for i in range(n_iter):
        primaldual = sqp(xi,c,lam)
        d          = primaldual[ :,:N ]
        lam_d      = primaldual[ :, N:N+M_in ]
        nu_d       = primaldual[ :,   N+M_in:N+M_in+M_eq ]
        xi         = xi+alpha*d
        lam        = lam + alpha*(lam_d - lam)
    return torch.cat([xi,lam,nu_d], dim=1)

""""""""""""""""""""""""""""""""""""""""""""""""



class EntropyKnapsackPGD():
    def __init__(self, N, C, stepsize = 0.001, n_iter=1000):
        super().__init__()

        self.alpha = stepsize
        eps = 1.0
        
        G = torch.cat(  (-torch.eye(N), torch.eye(N)),0  )
        h = torch.cat( (torch.zeros(N), torch.ones(N)) )
        A = torch.ones(N).unsqueeze(0)
        b = torch.Tensor([C])
        

        self.obj_grad = lambda x,c: obj_grad_fn(x,c,eps)
        self.obj_grad_batch = torch.func.vmap(lambda x,c: obj_grad_fn(x,c,eps), in_dims=(0,None))
        
        self.projection = get_projection_layer(A,b,G,h)
        
        self.fwd_solver = lambda c: fwd_solver(c, N, C, eps)
        self.fwd_solver_batch = lambda c: torch.stack([self.fwd_solver(c_i) for _, c_i in enumerate(c)])
        
        self.update_step  = lambda c,x: PGD(x,c,self.projection,self.obj_grad,self.alpha,n_iter=1).float()
        self.lmlLayer = FoldOptLayer(self.fwd_solver_batch, self.update_step, n_iter=n_iter, backprop_rule='FPI')

    def solve(self,c):
        return self.lmlLayer( c )





class EntropyKnapsackSQP():
    def __init__(self, N, C, stepsize=1.0, n_iter=1):
        super().__init__()

        self.alpha = stepsize
        eps = 1.0
        
        self.N = N
        self.M_in = 2*N
        self.M_eq = 1
        
        grad_f   = lambda x,c: -c + eps*(torch.log(x) + 1.0)
        hess_f   = lambda x,l,c:    eps*torch.diag( 1/x )
        grad_g   = lambda x: torch.cat( (-torch.eye(len(x)), torch.eye(len(x))),0 )
        grad_e   = lambda x: torch.ones(len(x)).unsqueeze(0)
        g        = lambda x: torch.cat( (-x,x-1) )
        e        = lambda x: (torch.sum(x) - C).unsqueeze(0)
        

        self.obj_grad = lambda x,c: obj_grad_fn(x,c,eps)
        self.obj_grad_batch = torch.func.vmap(lambda x,c: obj_grad_fn(x,c,eps), in_dims=(0,None))
        
        self.sqp = get_sqp_layer(g,e,grad_g,grad_e,grad_f,hess_f)
        
        self.fwd_solver = lambda c: fwd_solver(c,N,C,eps, primaldual=True)
        self.fwd_solver_batch = lambda c: torch.stack([self.fwd_solver(c_i) for _, c_i in enumerate(c)])
        
        self.update_step  = lambda c,x: SQP(x,c,self.sqp,self.N,self.M_in,self.M_eq,self.alpha,n_iter=1).float()
        self.lmlLayer = FoldOptLayer(self.fwd_solver_batch, self.update_step, n_iter=n_iter, backprop_rule='FPI')

    def solve(self,c):
        return self.lmlLayer( c )
        
        
