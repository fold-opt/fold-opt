

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from fold_opt.fold_opt import FoldOptLayer
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import cvxpy as cp

from multilabel_example.qpth_plus.qpth.qp import QPFunction



""" COPY THESE LINES """""""""""""""""""""""""""""
def obj_grad_fn(x,c,eps):
    return -c + eps*(torch.log(x) + 1.0 )

def fwd_solver(c, N, C, eps):
    # c = c.clone().cpu().numpy()
    # print(type(c), c.shape)
    
    x = cp.Variable(N)
    constraints = [  0<=x, x<=1, cp.sum(x) == C  ]
    problem = cp.Problem(cp.Maximize(c @ x + eps*cp.sum(cp.entr(x))), constraints)

    solution = problem.solve(solver=cp.ECOS)
    primal   = x.value
    dual = np.concatenate( tuple([np.array([constr.dual_value]).flatten() for constr in constraints]) )
    return torch.Tensor(primal), torch.Tensor(dual)

# def get_projection_layer(A,b,G,h):
#     N = A.shape[1] # number of variables
#     x = cp.Variable(N)
#     z = cp.Parameter(N)
#     constraints = [  A@x == b, G@x <= h   ]
#     problem = cp.Problem(cp.Minimize( cp.norm(x-z)**2  ),  constraints)

#     cvxlayer = CvxpyLayer(problem, parameters=[z], variables=[x])
#     cvxlayer_post = lambda z: cvxlayer(z)[0]
#     return cvxlayer_post

def get_projection_layer(A,b,G,h):
    N = A.shape[1] # number of variables  
    Q = torch.eye(N)
    return lambda x: QPFunction(verbose=-1,eps=1e-10,maxIter=1000)(2*Q.double(),-2*x.double(),G.double(),h.double(),A.double(),b.double())   

# All data inputs are expected to be torch Tensors
def PGD(x0,c,eps,projection,obj_grad,alpha=0.01,n_iter=1):
    xi = x0
    for i in range(n_iter):
        grads = obj_grad(xi,c)
        xi = projection(xi - alpha*grads)
    return xi
""""""""""""""""""""""""""""""""""""""""""""""""



class EntropyKnapsackPGD():
    def __init__(self, N, C, stepsize = 0.001, n_iter=1000):
        super().__init__()

        alpha = stepsize
        eps = 1.0
        
        G = torch.cat(  (-torch.eye(N), torch.eye(N)),0  )
        h = torch.cat( (torch.zeros(N), torch.ones(N)) )
        A = torch.ones(N).unsqueeze(0)
        b = torch.Tensor([C])
        

        self.obj_grad = lambda x,c: obj_grad_fn(x,c,eps)
        self.obj_grad_batch = torch.func.vmap(lambda x,c: obj_grad_fn(x,c,eps), in_dims=(0,None))
        
        self.projection = get_projection_layer(A,b,G,h)
        self.projection_batch = lambda x: torch.stack([self.projection(x_i) for _, x_i in enumerate(x)])
        
        self.fwd_solver = lambda c: fwd_solver(c, N, C, eps)[0]
        self.fwd_solver_batch = lambda c: torch.stack([self.fwd_solver(c_i) for _, c_i in enumerate(c)])
        
        self.update_step  = lambda c,x: PGD(x,c,eps,self.projection,self.obj_grad,alpha,n_iter=1).float()
        self.lmlLayer = FoldOptLayer(self.fwd_solver_batch, self.update_step, n_iter=n_iter, backprop_rule='FPI')

    def solve(self,c):
        return self.lmlLayer( c )





# class EntropyKnapsackSQP():
#     def __init__(self, N, C, stepsize=1.0, n_iter=1):
#         super().__init__()

#         self.alpha = stepsize
#         self.max_iter = n_iter
#         eps = 1.0
#         entro_knapsack_pd = cvxpy_models.EntropyNormKnapsackCVX(N,C,eps)
#         #p,d = entro_knapsack_pd.solve(coeffs)
#         entro_knapsack_pd_cat = lambda coeffs: torch.cat( entro_knapsack_pd.solve(coeffs.detach()) )
#         entro_knapsack_pd_batch = unrolled_ops.BatchModeWrapper(entro_knapsack_pd_cat)
#         self.entro_knapsack_pd_blank = unrolled_ops.BlankFunctionWrapper(N,entro_knapsack_pd_batch.apply)
#         self.N = N
#         self.M_in = 2*N
#         self.M_eq = 1
#         self.grad_f_ent   = lambda x,c: -c + eps*(torch.log(x) + 1.0)
#         self.hess_f_ent   = lambda x,l,c:    eps*torch.diag( 1/x )
#         self.grad_f = self.grad_f_ent
#         self.hess_L = self.hess_f_ent
#         self.g, self.e, self.grad_g, self.grad_e = diff_opt_tools.get_constraint_fns_knapsack(C)

#     def solve(self,c):
#         x_blank = self.entro_knapsack_pd_blank(c.double())
#         primal = x_blank[:,:self.N]
#         dual   = x_blank[:,self.N:self.N+self.M_in]  # There are 2N inequalities
#         return unrolled_ops.DiffSQP(c.double(), self.N, self.M_in, self.M_eq, self.g, self.e, self.grad_g, self.grad_e, self.grad_f, self.hess_L, self.alpha, self.max_iter, primal, dual ).float()





# class EntropyKnapsackFPGD():
#     def __init__(self, N, C, stepsize = 0.001, n_iter=1000):
#         super().__init__()

#         alpha = stepsize
#         self.n_iter = n_iter
#         eps = 1.0
#         entro_knapsack_pd = cvxpy_models.EntropyNormKnapsackCVX(N,C,eps)
#         entro_knapsack_pd_primal = lambda coeffs: entro_knapsack_pd.solve(coeffs.detach())[0]
#         entro_knapsack_pd_batch = unrolled_ops.BatchModeWrapper(entro_knapsack_pd_primal)
#         self.entro_knapsack_blank = unrolled_ops.BlankFunctionWrapper(N,entro_knapsack_pd_batch.apply)
#         self.fixedPtModule = fixedpt_ops.FixedPtDiff2()
#         grad_f_ent   = lambda x,c: -c + eps*(torch.log(x) + 1.0 )
#         A,b,G,h = diff_opt_tools.get_constraint_matrices_unwt_knapsack(N,C)
#         self.diff_step_op  = lambda clamb,xlamb: unrolled_ops.DiffPGD(clamb, N, grad_f_ent, G, h, A, b, alpha, 1, xlamb).float()


#     def solve(self,c):
#         x_blank = self.entro_knapsack_blank(c.double())
#         jacobian_x, x_star_step = fixedpt_ops.iterate_fwd(c, x_blank, self.diff_step_op)
#         x_pgd  = self.fixedPtModule.apply(c, x_blank, x_star_step, jacobian_x, self.n_iter)
#         return x_pgd
