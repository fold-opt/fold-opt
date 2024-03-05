# optimization tools / solvers based on cvxpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp

# Takes 1D Tensor (cost vector / no batch) and returns the primal and dual solutions (concatenated)
# maintains optimization variable and constraints in self
# Need a wrapper to use batch mode
class QuadNormKnapsackCVX():
    def __init__(self, N,C,eps):
        super(QuadNormKnapsackCVX, self).__init__()
        self.N = N
        self.C = C
        self.eps = eps
        self.x = cp.Variable(N)
        self.constraints = [  0<=self.x, self.x<=1, cp.sum(self.x) == C  ]  # fix this

    def solve(self, c):
        if len(c.shape)>1:
            input("Input to QuadNormKnapsackCVX must be Tensor of 1D")
            quit()
        problem  = cp.Problem(cp.Maximize(   c @ self.x - self.eps*cp.norm( self.x,p=2 )**2   ),  self.constraints)
        solution = problem.solve()
        primal   = self.x.value
        dual = np.concatenate( tuple([np.array([constr.dual_value]).flatten() for constr in self.constraints]) )
        return torch.Tensor(primal), torch.Tensor(dual)

class EntropyNormKnapsackCVX():
    def __init__(self, N,C,eps):
        super(EntropyNormKnapsackCVX, self).__init__()
        self.N = N
        self.C = C
        self.eps = eps
        self.x = cp.Variable(N)
        self.constraints = [  0<=self.x, self.x<=1, cp.sum(self.x) == C  ]  # fix this

    def solve(self, c):
        problem  = cp.Problem(cp.Maximize(   c @ self.x + self.eps*cp.sum(cp.entr( self.x ))    ),  self.constraints)
        solution = problem.solve()
        primal   = self.x.value


        dual = np.concatenate( tuple([np.array([constr.dual_value]).flatten() for constr in self.constraints]) )
        return torch.Tensor(primal), torch.Tensor(dual)
