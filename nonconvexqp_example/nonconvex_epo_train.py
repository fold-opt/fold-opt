import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import cvxpy as cp
import sys
import os
sys.path.append('..')
from fold_opt.fold_opt import FoldOptLayer
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from helper_qp import solve_nonconvexqp
sys.path.insert(1, os.path.join(sys.path[0], os.pardir))
#from solve_alm import solve_alm_nonconvexqp

#def obj_fn(self, X, Y):
#    return (0.5*(Y@self.Q)*Y + self.p*torch.sin(Y)).sum(dim=1)


""" COPY THESE LINES """""""""""""""""""""""""""""
def obj_fn_single(y,p,Q):
    return ( 0.5*(y@Q)@y + p@torch.sin(y) )

def get_projection_layer(A,b,G,h):
    N = A.shape[1] # number of variables
    x = cp.Variable(N)
    z = cp.Parameter(N)
    constraints = [ A@x == b, G@x <= h ]
    problem  = cp.Problem(cp.Minimize( cp.norm(x-z)**2  ),  constraints)

    qp_cvxlayer = CvxpyLayer(problem, parameters=[z], variables=[x])
    qp_cvxlayer_post = lambda z: qp_cvxlayer(z)[0]
    return qp_cvxlayer_post

# All data inputs are expected to be torch Tensors
def PGD(x0,p,Q,projection,obj_grad,alpha=0.01,n_iter=1):
    N = A.shape[1]
    xi = x0
    for i in range(n_iter):
        grads = obj_grad(xi,p,Q)
        xi = projection(xi - alpha*grads)
    return xi
""""""""""""""""""""""""""""""""""""""""""""""""



#nvar = 100 # the number of variables
#ntrain = 8334
#nvalid = 833
#ntest = 833
#nex = ntrain+nvalid+ntest # the total number of instances
#neq = 50 # the number of equality constraints
#nineq = 50 # the number of inequality constraints

nvar = 8 # the number of variables
ntrain = 10
nvalid = 10
ntest = 10
nex = ntrain+nvalid+ntest # the total number of instances
neq = 4 # the number of equality constraints
nineq = 4 # the number of inequality constraints

np.random.seed(1001)
Q = np.diag(np.random.random(nvar))
#p = np.random.random(nvar)
b = np.random.random(neq)
A = np.random.normal(loc=0, scale=1., size=(neq, nvar))
G = np.random.normal(loc=0, scale=1., size=(nineq, nvar))
h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)

trainX = np.random.uniform(-1, 1, size=(ntrain, nvar))
validX = np.random.uniform(-1, 1, size=(nvalid, nvar))
testX  = np.random.uniform(-1, 1, size=(ntest,  nvar))

trainY = solve_nonconvexqp(Q, trainX, A, G, h, b)[0]
validY = solve_nonconvexqp(Q, validX, A, G, h, b)[0]
testY  = solve_nonconvexqp(Q,  testX, A, G, h, b)[0]

trainX_tr = torch.Tensor(trainX)
Q_tr = torch.Tensor(Q)

""" COPY THESE LINES """""""""""""""""""""""""""""
obj_fn_batch   = torch.func.vmap(obj_fn_single, in_dims=(0,0,None))
obj_grad       = torch.func.grad(obj_fn_single)
obj_grad_batch = torch.func.vmap(obj_grad, in_dims=(0,0,None))

projection = get_projection_layer(A,b,G,h)

fwd_solver  = lambda p: torch.Tensor(solve_nonconvexqp(Q, np.array(p), A, G, h, b)[0])
update_step = lambda p,x: PGD(x,p,Q_tr,projection,obj_grad_batch,alpha=0.1,n_iter=1)
nonconvexQPlayer = FoldOptLayer(fwd_solver, update_step, n_iter=10, backprop_rule='FPI')
""""""""""""""""""""""""""""""""""""""""""""""""





predictor = torch.nn.Sequential( torch.nn.Linear( nvar, nvar ), torch.nn.ReLU(), torch.nn.BatchNorm1d( nvar ),
                                 torch.nn.Linear( nvar, nvar ) )


params   = trainX_tr
features = torch.rand(params.shape)

"""
Optional Pretraining of solution estimate
"""
batch = 50
eval_interval = 10
pretrain_L2_list = []
optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-2)
mse = torch.nn.MSELoss()
for epoch in range(1000):
    if epoch % eval_interval == 0:
        with torch.no_grad():
            "This is where testing routines would go"

    idx = torch.randperm(params.shape[0])[:batch]
    features_true = features[idx]
    params_true   =   params[idx]

    """ COPY THESE LINES """""""""""""""""""""""""""
    params_pred = predictor( features_true )
    optsol_batch = nonconvexQPlayer( params_pred )
    loss = obj_fn_batch(optsol_batch,params_true,Q_tr).sum()
    """"""""""""""""""""""""""""""""""""""""""""""""

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("loss = {}".format(loss.item()))
