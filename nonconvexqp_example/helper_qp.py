"""
    This code is based on the official DC3 implementation in https://github.com/locuslab/DC3
"""
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

import operator
from functools import reduce
import numpy as np
import osqp
import cyipopt     # JK to suppress cyiopt not installed - version conflict (requires python 3.11, pytorch not compatible)
from scipy.sparse import csc_matrix

import time

###################################################################
# CONVEX QP PROBLEM
###################################################################
def solve_convexqp(Q, p, A, G, h, X, tol=1e-4):
    print('running osqp')
    ydim = Q.shape[0]
    my_A = np.vstack([A, G])
    total_time = 0
    Y = []
    for Xi in X:
        solver = osqp.OSQP()
        my_l = np.hstack([Xi, -np.ones(h.shape[0]) * np.inf])
        my_u = np.hstack([Xi, h])
        solver.setup(P=csc_matrix(Q), q=p, A=csc_matrix(my_A), l=my_l, u=my_u, verbose=False, eps_prim_inf=tol)
        start_time = time.time()
        results = solver.solve()
        end_time = time.time()

        total_time += (end_time - start_time)
        if results.info.status == 'solved':
            Y.append(results.x)
        else:
            Y.append(np.ones(ydim) * np.nan)

        sols = np.array(Y)

    return sols, total_time, total_time/X.shape[0]


class ConvexQPProblem:
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
    """
    def __init__(self, Q, p, A, G, h, trainX, validX, testX, trainY, validY, testY, device, testY_ALM=None):
        self.device = device
        self.Q = torch.tensor(Q).to(device)
        self.p = torch.tensor(p).to(device)
        self.A = torch.tensor(A).to(device)
        self.G = torch.tensor(G).to(device)
        self.h = torch.tensor(h).to(device)

        self.xdim = trainX.shape[1]
        self.ydim = Q.shape[0]
        self.neq = A.shape[0]
        self.nineq = G.shape[0]

        self.nknowns = 0 # only for DC3

        self.trainX = torch.tensor(trainX).to(device)
        self.validX = torch.tensor(validX).to(device)
        self.testX = torch.tensor(testX).to(device)
        self.trainY = torch.tensor(trainY).to(device)
        self.validY = torch.tensor(validY).to(device)
        self.testY = torch.tensor(testY).to(device)
        if testY_ALM is not None:
            self.testY_ALM = torch.tensor(testY_ALM).to(device)
        else:
            self.testY_ALM = None

        self.nex = trainX.shape[0] + validX.shape[0] + testX.shape[0]
        assert trainX.shape[0] == trainY.shape[0]
        assert validX.shape[0] == validY.shape[0]
        assert testX.shape[0] == testY.shape[0]
        if testY_ALM is not None:
            assert testY.shape[0] == testY_ALM.shape[0]

        # for DC3
        det = 0
        i = 0
        while abs(det) < 0.0001 and i < 100:
            self.partial_vars = np.random.choice(self.ydim, self.ydim - self.neq, replace=False)
            self.other_vars = np.setdiff1d( np.arange(self.ydim), self.partial_vars)
            det = torch.det(self.A[:, self.other_vars])
            i += 1
        if i == 100:
            raise Exception
        else:
            self.A_partial = self.A[:, self.partial_vars]
            self.A_other_inv = torch.inverse(self.A[:, self.other_vars])

    def __str__(self):
        return 'ConvexQPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.nex)
        )

    @property
    def obj_scaler(self):
        return 1.

    def obj_fn(self, X, Y):
        return (0.5*(Y@self.Q)*Y + self.p*Y).sum(dim=1)

    def opt_gap(self, X, Y, Ygt):
        obj_app = self.obj_fn(X, Y)
        obj_gt = self.obj_fn(X, Ygt)
        return (obj_app-obj_gt).abs()/obj_gt.abs()

    def eq_resid(self, X, Y):
        return Y@self.A.T - X

    def ineq_resid(self, X, Y):
        return Y@self.G.T - self.h

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y):
        ineq_dist = self.ineq_dist(X, Y)
        return 2*ineq_dist@self.G

    def ineq_partial_grad(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self.A_other_inv @ self.A_partial)
        h_effective = self.h - (X @ self.A_other_inv.T) @ self.G[:, self.other_vars].T
        grad = 2 * torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ G_effective
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = grad
        Y[:, self.other_vars] = - (grad @ self.A_partial.T) @ self.A_other_inv.T
        return Y

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self.A_partial.T) @ self.A_other_inv.T
        return Y


###################################################################
# NONCONVEX VARIANT OF THE QP PROBLEM
###################################################################
def solve_nonconvexqp(Q, p, A, G, h, X, tol=1e-4):
    Y = []
    total_time = 0
    for ii, pi in enumerate(p):
        print(ii,end='\r')
        y0 = np.linalg.pinv(A)@X  # feasible initial point

        # upper and lower bounds on variables
        lb = -np.infty * np.ones(y0.shape)
        ub = np.infty * np.ones(y0.shape)

        # upper and lower bounds on constraints
        cl = np.hstack([X, -np.inf * np.ones(G.shape[0])])
        cu = np.hstack([X, h])

        nlp = cyipopt.problem(
                    n=len(y0),
                    m=len(cl),
                    problem_obj=nonconvex_ipopt(Q, pi, A, G),
                    lb=lb,
                    ub=ub,
                    cl=cl,
                    cu=cu)

        nlp.addOption('tol', tol)
        nlp.addOption('print_level', 0)

        start_time = time.time()
        y, info = nlp.solve(y0)
        end_time = time.time()
        Y.append(y)
        total_time += (end_time - start_time)

    return np.array(Y), total_time, total_time/X.shape[0]
"""
def solve_nonconvexqp(Q, p, A, G, h, X, tol=1e-4):
    Y = []
    total_time = 0
    for ii, Xi in enumerate(X):
        print(ii,end='\r')
        y0 = np.linalg.pinv(A)@Xi  # feasible initial point

        # upper and lower bounds on variables
        lb = -np.infty * np.ones(y0.shape)
        ub = np.infty * np.ones(y0.shape)

        # upper and lower bounds on constraints
        cl = np.hstack([Xi, -np.inf * np.ones(G.shape[0])])
        cu = np.hstack([Xi, h])

        nlp = cyipopt.problem(
                    n=len(y0),
                    m=len(cl),
                    problem_obj=nonconvex_ipopt(Q, p, A, G),
                    lb=lb,
                    ub=ub,
                    cl=cl,
                    cu=cu)

        nlp.addOption('tol', tol)
        nlp.addOption('print_level', 0)

        start_time = time.time()
        y, info = nlp.solve(y0)
        end_time = time.time()
        Y.append(y)
        total_time += (end_time - start_time)

    return np.array(Y), total_time, total_time/X.shape[0]
"""


class nonconvex_ipopt(object):
    def __init__(self, Q, p, A, G):
        self.Q = Q
        self.p = p
        self.A = A
        self.G = G
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p@np.sin(y)

    def gradient(self, y):
        return self.Q@y + (self.p * np.cos(y))

    def constraints(self, y):
        return np.hstack([self.A@y, self.G@y])

    def jacobian(self, y):
        return np.concatenate([self.A.flatten(), self.G.flatten()])


class NonconvexQPProblem:
    """
        minimize_y 1/2 * y^T Q y + p^T sin(y)
        s.t.       Ay =  x
                   Gy <= h
    """
    def __init__(self, Q, p, A, G, h, trainX, validX, testX, trainY, validY, testY, device, testY_ALM=None):
        self.device = device
        self.Q = torch.tensor(Q).to(device)
        self.p = torch.tensor(p).to(device)
        self.A = torch.tensor(A).to(device)
        self.G = torch.tensor(G).to(device)
        self.h = torch.tensor(h).to(device)

        self.xdim = trainX.shape[1]
        self.ydim = Q.shape[0]
        self.neq = A.shape[0]
        self.nineq = G.shape[0]

        self.nknowns = 0 # only for DC3

        self.trainX = torch.tensor(trainX).to(device)
        self.validX = torch.tensor(validX).to(device)
        self.testX = torch.tensor(testX).to(device)
        self.trainY = torch.tensor(trainY).to(device)
        self.validY = torch.tensor(validY).to(device)
        self.testY = torch.tensor(testY).to(device)
        if testY_ALM is not None:
            self.testY_ALM = torch.tensor(testY_ALM).to(device)
        else:
            self.testY_ALM = None

        self.nex = trainX.shape[0] + validX.shape[0] + testX.shape[0]
        assert trainX.shape[0] == trainY.shape[0]
        assert validX.shape[0] == validY.shape[0]
        assert testX.shape[0] == testY.shape[0]
        if testY_ALM is not None:
            assert testY.shape[0] == testY_ALM.shape[0]

        det = 0
        i = 0
        while abs(det) < 0.0001 and i < 100:
            self.partial_vars = np.random.choice(self.ydim, self.ydim - self.neq, replace=False)
            self.other_vars = np.setdiff1d(np.arange(self.ydim), self.partial_vars)
            det = torch.det(self.A[:, self.other_vars])
            i += 1
        if i == 100:
            raise Exception
        else:
            self.A_partial = self.A[:, self.partial_vars]
            self.A_other_inv = torch.inverse(self.A[:, self.other_vars])
            self.M = 2 * (self.G[:, self.partial_vars] -
                          self.G[:, self.other_vars] @ (self.A_other_inv @ self.A_partial))

    def __str__(self):
        return 'NonconvexQPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.nex)
        )

    @property
    def obj_scaler(self):
        return 1.

    def obj_fn(self, X, Y):
        return (0.5*(Y@self.Q)*Y + self.p*torch.sin(Y)).sum(dim=1)

    def opt_gap(self, X, Y, Ygt):
        obj_app = self.obj_fn(X, Y)
        obj_gt = self.obj_fn(X, Ygt)
        return (obj_app-obj_gt).abs()/obj_gt.abs()

    def eq_resid(self, X, Y):
        return X - Y@self.A.T

    def ineq_resid(self, X, Y):
        return Y@self.G.T - self.h

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y):
        return 2 * torch.clamp(Y@self.G.T - self.h, 0) @ self.G

    def ineq_partial_grad(self, X, Y):
        grad = torch.clamp(Y@self.G.T - self.h, 0) @ self.M
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = grad
        Y[:, self.other_vars] = - (grad @ self.A_partial.T) @ self.A_other_inv.T
        return Y

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self.A_partial.T) @ self.A_other_inv.T
        return Y


def load_qp_data(args, current_path, device):
    probtype = args['probtype']
    filename = "%s_var%d_ineq%d_eq%d_ex%d.npz"%(probtype, args['nvar'], args['nineq'], args['neq'], args['nex'])
    filepath = current_path/"datasets"/probtype/filename
    loaded_data = np.load(filepath)
    if probtype == 'convexqp':
        data = ConvexQPProblem(device=device, **loaded_data)
    elif probtype == 'nonconvexqp':
        data = NonconvexQPProblem(device=device, **loaded_data)
    else:
        raise NotImplementedError
    return data


###################################################################
# NEURAL NETWORKS
###################################################################

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class NNPrimalSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim]
        layer_sizes += self._args['nlayer']*[self._args['hiddensize']]
        layers = reduce(operator.add, [[nn.Linear(a,b), nn.ReLU()] for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        output_dim = data.ydim
        layers += [nn.Linear(layer_sizes[-1], output_dim)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(DEVICE)
        return self.net(x)


class NNDualSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim]
        layer_sizes += self._args['nlayer']*[self._args['hiddensize']]
        layers = reduce(operator.add, [[nn.Linear(a,b), nn.ReLU()] for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

        self.lag_eq = nn.Linear(layer_sizes[-1],data.neq)
        self.lag_ineq = nn.Linear(layer_sizes[-1],data.nineq)

        # initialize to output zeros at the beginning
        nn.init.zeros_(self.lag_eq.weight); nn.init.zeros_(self.lag_eq.bias)
        nn.init.zeros_(self.lag_ineq.weight); nn.init.zeros_(self.lag_ineq.bias)


    def forward(self, x):
        x = x.to(DEVICE)
        out = self.net(x)
        return self.lag_eq(out), self.lag_ineq(out)
