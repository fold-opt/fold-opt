import numpy as np
import math
import pickle as pkl
import argparse
import cvxpy as cp



# cvxpy versions
def get_markowitz_constraints_cvx(n,p,tau,L):
    e = np.ones(n)
    COV = np.matmul(L,L.T) + np.eye(n)*(0.01*tau)**2
    w_ = e/10
    gamma = 2.25 * np.matmul( np.matmul(w_,COV), w_ )
    x = cp.Variable(n)
    constraints = [  x >= 0, e @ x <= 1 ]
    return constraints, x


def solve_markowitz_cvx(constraints,variables,c):
    x = variables
    prob = cp.Problem(cp.Maximize( c @ x ),   # (1/2)*cp.quad_form(x, P) +
                      constraints)
    prob.solve()

    return np.array(x.value)